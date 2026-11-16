import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

BOARD_PATH = Path("outputs") / "comparison_board.csv"
OUTPUT_PATH = Path("outputs") / "mcdm_ranking.csv"

# Tiêu chí dùng để xếp hạng và hướng tối ưu của nó:
#   "benefit" = càng lớn càng tốt, "cost" = càng nhỏ càng tốt.
# Không dùng accuracy/precision/recall vì F1 đã được tính từ precision và recall;
# đưa tất cả vào sẽ đếm "chất lượng dự báo" nhiều lần.
CRITERIA = {
    "f1": "benefit",
    "roc_auc": "benefit",
    "fit_time_s": "cost",
    "predict_1000_ms": "cost",
    "model_size_mb": "cost",
}

# Phán đoán so sánh cặp theo thang Saaty 1-9.
# (A, B): v  nghĩa là  "A quan trọng hơn B v lần"; v < 1 nghĩa là B quan trọng hơn A.
# Mỗi cặp chỉ điền một lần, chiều ngược lại được suy ra bằng nghịch đảo.
PAIRWISE = {
    ("f1", "roc_auc"): 2,
    ("f1", "fit_time_s"): 5,
    ("f1", "predict_1000_ms"): 4,
    ("f1", "model_size_mb"): 6,
    ("roc_auc", "fit_time_s"): 4,
    ("roc_auc", "predict_1000_ms"): 3,
    ("roc_auc", "model_size_mb"): 5,
    ("fit_time_s", "predict_1000_ms"): 1 / 2,
    ("fit_time_s", "model_size_mb"): 2,
    ("predict_1000_ms", "model_size_mb"): 3,
}

# Random Index của Saaty: CI trung bình của ma trận so sánh cặp điền ngẫu nhiên, theo kích thước n.
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
CR_LIMIT = 0.10

NORMALIZATION = "vector"  # "minmax" hoặc "vector"
EXAMPLE_ALGORITHM = "XGBoost"  # thuật toán được dùng để minh họa cách tính từng ô
OLD_RULE_F1_TOLERANCE = 0.01  # quy tắc chọn thủ công cũ trong train_model.py

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


# ----------------------------------------------------------------------------- in ấn
def title(text):
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def note(*lines):
    for line in lines:
        print(f"  {line}")


def show(table, decimals=4):
    print()
    print(table.round(decimals).to_string())
    print()


def example_row(index):
    return EXAMPLE_ALGORITHM if EXAMPLE_ALGORITHM in index else index[0]


# ----------------------------------------------------------------------------- tính toán thuần
def build_pairwise_matrix():
    names = list(CRITERIA)
    for a, b in PAIRWISE:
        if a not in CRITERIA or b not in CRITERIA:
            raise ValueError(f"PAIRWISE uses a criterion that is not in CRITERIA: ({a}, {b})")
        if a == b:
            raise ValueError(f"PAIRWISE must not compare a criterion with itself: ({a}, {b})")
    given = [frozenset(pair) for pair in PAIRWISE]
    if len(set(given)) != len(given):
        raise ValueError("PAIRWISE has a pair entered twice (both (A, B) and (B, A)).")
    missing = {frozenset(pair) for pair in combinations(names, 2)} - set(given)
    if missing:
        raise ValueError(f"PAIRWISE is missing these pairs: {[tuple(pair) for pair in missing]}")

    matrix = pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)
    for (a, b), value in PAIRWISE.items():
        matrix.loc[a, b] = value
        matrix.loc[b, a] = 1 / value
    return matrix


def ahp_weights(matrix):
    normalized = matrix / matrix.sum(axis=0)
    return normalized, normalized.mean(axis=1)


def consistency(matrix, weights):
    n = len(matrix)
    weighted_sum = matrix.values @ weights.values
    lambdas = weighted_sum / weights.values
    lambda_max = lambdas.mean()
    ci = (lambda_max - n) / (n - 1)
    ri = RANDOM_INDEX[n]
    cr = ci / ri if ri else 0.0
    return weighted_sum, lambdas, lambda_max, ci, ri, cr


def normalize(X, method):
    if method == "minmax":
        span = (X.max() - X.min()).replace(0, 1)
        return (X - X.min()) / span
    if method == "vector":
        return X / np.sqrt((X**2).sum())
    raise ValueError(f"Invalid NORMALIZATION: {method}")


def ideal_points(V):
    benefit = [CRITERIA[c] == "benefit" for c in V.columns]
    best = pd.Series(np.where(benefit, V.max(), V.min()), index=V.columns)
    worst = pd.Series(np.where(benefit, V.min(), V.max()), index=V.columns)
    return best, worst


def topsis(X, weights, method):
    V = normalize(X, method) * weights
    best, worst = ideal_points(V)
    d_plus = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - worst) ** 2).sum(axis=1))
    return d_plus, d_minus, d_minus / (d_plus + d_minus)


# ----------------------------------------------------------------------------- các bước có giải thích
def step0_decision_matrix():
    title("STEP 0 - DECISION MATRIX")
    board = pd.read_csv(BOARD_PATH, index_col="algorithm")
    X = board[list(CRITERIA)]
    note(
        f"Read from {BOARD_PATH}. Each ROW is an alternative (algorithm), each COLUMN is a criterion.",
        f"Size: {X.shape[0]} alternatives x {X.shape[1]} criteria.",
    )
    show(X)
    note("Optimization direction of each criterion:")
    for criterion, kind in CRITERIA.items():
        meaning = "HIGHER is better" if kind == "benefit" else "LOWER is better"
        note(f"    {criterion:<16} {kind:<8} -> {meaning}")
    note(
        "",
        "Problem: the columns have different units (F1 in [0, 1], time in seconds/ms, size in MB),",
        "so they cannot simply be added together.",
        "  - AHP (steps 1-3)    answers: how important (in percent) is each criterion?",
        "  - TOPSIS (steps 4-8) answers: combining all criteria, which algorithm is best?",
    )
    return board, X


def step1_pairwise_matrix():
    title("STEP 1 (AHP) - PAIRWISE COMPARISON MATRIX")
    matrix = build_pairwise_matrix()
    note(
        "Saaty scale: 1 = equal, 3 = moderately more, 5 = strongly more, 7 = very strongly more, 9 = extremely more",
        "             (2, 4, 6, 8 are intermediate values).",
        "Cell [row i, column j] = how many times criterion i is more important than criterion j.",
        "Two rules: diagonal = 1 (compared with itself), and a[j, i] = 1 / a[i, j] (reciprocal).",
        "",
        "Judgments entered (PAIRWISE config at the top of the file):",
    )
    for (a, b), value in PAIRWISE.items():
        if value >= 1:
            note(f"    {a} is {value:g}x more important than {b}  ->  a[{a}, {b}] = {value:g},  a[{b}, {a}] = 1/{value:g} = {1 / value:.4f}")
        else:
            note(f"    {b} is {1 / value:g}x more important than {a}  ->  a[{a}, {b}] = 1/{1 / value:g} = {value:.4f},  a[{b}, {a}] = {1 / value:g}")
    show(matrix, 3)
    return matrix


def step2_ahp_weights(matrix):
    title("STEP 2 (AHP) - DERIVE WEIGHTS FROM THE PAIRWISE COMPARISON MATRIX")
    names = list(matrix.index)
    col_sum = matrix.sum(axis=0)
    normalized, weights = ahp_weights(matrix)

    note("2a. Sum each column of the pairwise comparison matrix:")
    show(col_sum.to_frame("column sum").T)

    row, col = names[0], names[2]
    note(
        "2b. Normalize: divide each cell by its column sum  ->  every column then sums to 1.",
        f"    Example cell [{row}, {col}] = {matrix.loc[row, col]:.4f} / {col_sum[col]:.4f} = {normalized.loc[row, col]:.4f}",
    )
    show(normalized)

    terms = " + ".join(f"{value:.4f}" for value in normalized.loc[row])
    note(
        "2c. Weight of a criterion = AVERAGE of its row in the normalized matrix.",
        f"    Example w[{row}] = ({terms}) / {len(names)} = {weights[row]:.4f}",
    )
    table = pd.DataFrame({"weight w": weights, "percent": weights * 100})
    show(table.sort_values("weight w", ascending=False))
    note(f"Sum of weights = {weights.sum():.4f} (always 1).")
    return weights


def step3_consistency(matrix, weights):
    title("STEP 3 (AHP) - CONSISTENCY CHECK (consistency ratio, CR)")
    names = list(matrix.index)
    n = len(names)
    a, b, c = names[0], names[1], names[2]
    ab, bc, ac = matrix.loc[a, b], matrix.loc[b, c], matrix.loc[a, c]
    note(
        "Idea: judgments should be transitive. Example from the matrix in step 1:",
        f"  {a} is {ab:.3g}x {b}, and {b} is {bc:.3g}x {c}",
        f"  -> logically {a} should be {ab:.3g} x {bc:.3g} = {ab * bc:.3g}x {c}, but {ac:.3g} was entered.",
        "Small deviations are normal. CR measures the overall 'self-contradiction'; CR < 0.10 is acceptable.",
    )

    weighted_sum, lambdas, lambda_max, ci, ri, cr = consistency(matrix, weights)
    terms = " + ".join(f"{matrix.iloc[0, j]:.3f}x{weights.iloc[j]:.4f}" for j in range(n))
    note(
        "",
        "3a. Multiply matrix A by the weight vector w, then divide each element by w_i:",
        f"    Example (A*w)[{a}] = {terms} = {weighted_sum[0]:.4f}",
        f"          lambda[{a}] = {weighted_sum[0]:.4f} / {weights.iloc[0]:.4f} = {lambdas[0]:.4f}",
    )
    show(pd.DataFrame({"(A*w)_i": weighted_sum, "w_i": weights, "lambda_i": lambdas}, index=names))

    note(
        f"3b. lambda_max = average of lambda_i = {lambda_max:.4f}",
        f"    (for a perfectly consistent matrix lambda_max equals n = {n})",
        f"3c. CI = (lambda_max - n) / (n - 1) = ({lambda_max:.4f} - {n}) / {n - 1} = {ci:.4f}",
        f"3d. RI = {ri} (Saaty table for n = {n}: average CI of a randomly filled matrix)",
        f"3e. CR = CI / RI = {ci:.4f} / {ri} = {cr:.4f}",
    )
    if cr >= CR_LIMIT:
        note("", f"CONCLUSION: CR = {cr:.4f} >= {CR_LIMIT} -> judgments are too inconsistent, NOT usable.")
        raise SystemExit("Please revise PAIRWISE at the top of the file and run again.")
    note("", f"CONCLUSION: CR = {cr:.4f} < {CR_LIMIT} -> judgments are consistent enough, these weights can be used.")

    values, vectors = np.linalg.eig(matrix.values)
    principal = np.abs(vectors[:, np.argmax(values.real)].real)
    eigen_weights = pd.Series(principal / principal.sum(), index=names)
    note(
        "",
        "Cross-check: Saaty's 'standard' method takes the principal eigenvector of matrix A.",
        "The row-average method in step 2 is a hand-computable approximation; the two nearly match:",
    )
    show(pd.DataFrame({"row average (used)": weights, "eigenvector": eigen_weights, "difference": weights - eigen_weights}))


def step4_normalize(X):
    title(f"STEP 4 (TOPSIS) - NORMALIZE THE DECISION MATRIX (method: {NORMALIZATION})")
    R = normalize(X, NORMALIZATION)
    example = example_row(X.index)

    if NORMALIZATION == "minmax":
        note(
            "Min-max formula:  r = (x - column min) / (column max - column min)",
            "After this step every column is unitless: column minimum = 0, column maximum = 1.",
        )
        show(pd.DataFrame({"min": X.min(), "max": X.max(), "max - min": X.max() - X.min()}).T)
        note(f"Example calculation for row {example}:")
        for c in X.columns:
            x, low, high = X.loc[example, c], X[c].min(), X[c].max()
            note(f"    {c:<16} ({x:.4f} - {low:.4f}) / ({high:.4f} - {low:.4f}) = {R.loc[example, c]:.4f}")
    else:
        note("Vector formula:  r = x / sqrt(sum of squares of the whole column)")
        show(np.sqrt((X**2).sum()).to_frame("sqrt(sum x^2)").T)
        note(f"Example calculation for row {example}:")
        for c in X.columns:
            x, length = X.loc[example, c], np.sqrt((X[c] ** 2).sum())
            note(f"    {c:<16} {x:.4f} / {length:.4f} = {R.loc[example, c]:.4f}")
    show(R)
    note(
        "Note: 'cost' columns keep their original direction here (with min-max: 0 = smallest = best).",
        "The better/worse direction is handled in step 6 when choosing the ideal point.",
        "",
        "Why does normalization matter? Compare the spread (max - min) of each column under both methods:",
    )
    by_vector, by_minmax = normalize(X, "vector"), normalize(X, "minmax")
    show(
        pd.DataFrame(
            {
                "spread (vector)": by_vector.max() - by_vector.min(),
                "spread (min-max)": by_minmax.max() - by_minmax.min(),
            }
        )
    )
    note(
        "With vector normalization, columns whose values are close together (like f1) get squeezed into a",
        "very narrow range, so even with a large AHP weight they barely affect the result. Min-max stretches",
        "every column to the same spread = 1, so the AHP weights act exactly as declared. The trade-off: min-max",
        "also magnifies tiny differences (e.g. roc_auc differs by only ~0.04 but is stretched to 0 -> 1).",
        "Step 9 compares both methods.",
    )
    return R


def step5_weighted(R, weights):
    title("STEP 5 (TOPSIS) - APPLY AHP WEIGHTS TO THE NORMALIZED MATRIX")
    V = R * weights
    example = example_row(R.index)
    note("Formula: v[i, j] = w[j] x r[i, j]   (w comes from step 2)")
    show(weights.to_frame("w").T)
    note(f"Example calculation for row {example}:")
    for c in R.columns:
        note(f"    {c:<16} {weights[c]:.4f} x {R.loc[example, c]:.4f} = {V.loc[example, c]:.4f}")
    show(V)
    note(
        "A criterion with a larger weight becomes 'longer' in the space, so distances along it",
        "influence the result more. With min-max, each column's maximum now equals its weight.",
    )
    return V


def step6_ideal_points(V):
    title("STEP 6 (TOPSIS) - DETERMINE THE IDEAL POINT A+ AND THE ANTI-IDEAL POINT A-")
    best, worst = ideal_points(V)
    note(
        "For benefit criteria: A+ = column MAXIMUM, A- = column MINIMUM.",
        "For cost criteria:    A+ = column MINIMUM, A- = column MAXIMUM.",
    )
    rows = []
    for c in V.columns:
        is_benefit = CRITERIA[c] == "benefit"
        rows.append(
            {
                "criterion": c,
                "type": CRITERIA[c],
                "A+": best[c],
                "A+ taken from": V[c].idxmax() if is_benefit else V[c].idxmin(),
                "A-": worst[c],
                "A- taken from": V[c].idxmin() if is_benefit else V[c].idxmax(),
            }
        )
    show(pd.DataFrame(rows).set_index("criterion"))
    note("Written compactly as 2 vectors:")
    show(pd.DataFrame([best, worst], index=["A+ (ideal)", "A- (anti-ideal)"]))
    note(
        "A+ is a 'virtual algorithm' assembled from the best value of every algorithm; it usually does not exist.",
        "TOPSIS looks for the real algorithm that is CLOSEST to A+ and at the same time FARTHEST from A-.",
    )
    return best, worst


def step7_distances(V, best, worst):
    title("STEP 7 (TOPSIS) - EUCLIDEAN DISTANCE TO A+ (d+) AND TO A- (d-)")
    example = example_row(V.index)
    squared_plus = (V - best) ** 2
    squared_minus = (V - worst) ** 2
    d_plus = np.sqrt(squared_plus.sum(axis=1))
    d_minus = np.sqrt(squared_minus.sum(axis=1))

    for label, squared, distance, point in [
        ("d+", squared_plus, d_plus, "A+"),
        ("d-", squared_minus, d_minus, "A-"),
    ]:
        note(f"Squared difference of each cell to {point}:  (v[i, j] - {point}[j])^2")
        show(squared, 6)
        terms = " + ".join(f"{value:.6f}" for value in squared.loc[example])
        note(
            f"{label} = square root of the row sum. Example {example}:",
            f"    {label} = sqrt({terms})",
            f"       = sqrt({squared.loc[example].sum():.6f}) = {distance[example]:.4f}",
            "",
        )
    show(pd.DataFrame({"d+ (to ideal)": d_plus, "d- (to anti-ideal)": d_minus}))
    note("Smaller d+ is better (close to the ideal). Larger d- is better (far from the worst point).")
    return d_plus, d_minus, squared_plus


def step8_closeness(d_plus, d_minus):
    title("STEP 8 (TOPSIS) - CLOSENESS COEFFICIENT C AND RANKING")
    closeness = d_minus / (d_plus + d_minus)
    note(
        "Formula: C = d- / (d+ + d-),  C lies in [0, 1].",
        "  C = 1  -> the algorithm coincides with A+ (perfect)",
        "  C = 0  -> the algorithm coincides with A- (worst in every respect)",
        "",
    )
    for name in closeness.sort_values(ascending=False).index:
        note(f"    C[{name:<18}] = {d_minus[name]:.4f} / ({d_plus[name]:.4f} + {d_minus[name]:.4f}) = {closeness[name]:.4f}")
    result = pd.DataFrame({"d+": d_plus, "d-": d_minus, "C": closeness}).sort_values("C", ascending=False)
    result.insert(0, "rank", range(1, len(result) + 1))
    show(result)
    return result


def step9_sensitivity(X, weights, result):
    title("STEP 9 - SENSITIVITY ANALYSIS: DOES THE RESULT CHANGE UNDER DIFFERENT ASSUMPTIONS?")
    note(
        "The AHP weights and the normalization method are both choices made by the analyst. A trustworthy",
        "conclusion should not collapse when those choices change slightly. Re-running TOPSIS with other assumptions:",
    )
    benefit = pd.Series([CRITERIA[c] == "benefit" for c in X.columns], index=X.columns)

    def split(benefit_share):
        return benefit.map({True: benefit_share / benefit.sum(), False: (1 - benefit_share) / (~benefit).sum()})

    other = "vector" if NORMALIZATION == "minmax" else "minmax"
    scenarios = [
        (f"AHP + {NORMALIZATION} (main result)", weights, NORMALIZATION),
        (f"AHP + {other}", weights, other),
        (f"equal weights + {NORMALIZATION}", pd.Series(1 / len(X.columns), index=X.columns), NORMALIZATION),
        (f"quality-first 70/30 + {NORMALIZATION}", split(0.7), NORMALIZATION),
        (f"cost-first 30/70 + {NORMALIZATION}", split(0.3), NORMALIZATION),
    ]
    rows = []
    for label, scenario_weights, method in scenarios:
        closeness = topsis(X, scenario_weights, method)[2].sort_values(ascending=False)
        rows.append(
            {
                "scenario": label,
                "rank 1": closeness.index[0],
                "full order (C score)": "  >  ".join(f"{k} ({v:.3f})" for k, v in closeness.items()),
            }
        )
    table = pd.DataFrame(rows).set_index("scenario")
    print()
    print(table.to_string())
    print()

    winner = result.index[0]
    wins = (table["rank 1"] == winner).sum()
    note(f"{winner} ranks first in {wins}/{len(table)} scenarios.")
    if wins == len(table):
        note("-> ROBUST conclusion: the winner does not change regardless of weights or normalization.")
    else:
        others = ", ".join(
            f"{name} ({count} scenario(s))" for name, count in table["rank 1"].value_counts().items() if name != winner
        )
        note(
            f"-> ASSUMPTION-DEPENDENT conclusion: in the remaining scenarios, rank 1 goes to {others}.",
            "   When reporting, state the chosen assumptions (weights + normalization) and the reasons.",
        )


def step10_final_ranking(board, result, squared_plus):
    title("FINAL RESULT - RANKING OF THE 4 ALGORITHMS BY AHP + TOPSIS")
    final = result.join(board[list(CRITERIA)])
    final["main weakness"] = squared_plus.idxmax(axis=1)
    final.index.name = "algorithm"
    show(final)
    final.round(4).to_csv(OUTPUT_PATH)
    note(
        f"Table saved to {OUTPUT_PATH}",
        "",
        "How to read the table:",
        "  C             : overall TOPSIS score, closer to 1 is better -> determines the rank",
        "  d+ / d-       : distance to the ideal / anti-ideal algorithm",
        "  main weakness : criterion that pulls the algorithm farthest from A+ (largest contribution to d+)",
    )

    winner, runner_up = result.index[0], result.index[1]
    gap = result.loc[winner, "C"] - result.loc[runner_up, "C"]
    note(
        "",
        f"SELECTED ALGORITHM: {winner}  (C = {result.loc[winner, 'C']:.4f})",
        f"Runner-up is {runner_up} (C = {result.loc[runner_up, 'C']:.4f}), {gap:.4f} points behind.",
    )

    top_f1 = board["f1"].max()
    candidates = board[board["f1"] >= top_f1 - OLD_RULE_F1_TOLERANCE]
    old_winner = candidates.sort_values(["model_size_mb", "predict_1000_ms"]).index[0]
    note("", f"Compared with the old manual rule in train_model.py: it picks {old_winner}.")
    if old_winner == winner:
        note("-> Both approaches agree, but now there is a score, a full ranking and a quantitative justification.")
    else:
        note(
            "-> The two approaches DISAGREE. The old rule only looks at F1 (with an arbitrary 0.01 tolerance), then",
            "   picks the smallest model, ignoring roc_auc and treating any F1 gap below 0.01 as a tie. AHP + TOPSIS",
            "   weighs all criteria together using weights that passed the consistency check.",
        )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not BOARD_PATH.exists():
        raise SystemExit(f"{BOARD_PATH} not found. Run train_model.py first.")

    board, X = step0_decision_matrix()
    matrix = step1_pairwise_matrix()
    weights = step2_ahp_weights(matrix)
    step3_consistency(matrix, weights)
    R = step4_normalize(X)
    V = step5_weighted(R, weights)
    best, worst = step6_ideal_points(V)
    d_plus, d_minus, squared_plus = step7_distances(V, best, worst)
    result = step8_closeness(d_plus, d_minus)
    step9_sensitivity(X, weights, result)
    step10_final_ranking(board, result, squared_plus)


if __name__ == "__main__":
    main()
