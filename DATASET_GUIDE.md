# Your First ML Model: "Will it rain in the next hour?"

This guide covers parts 1–3 of the project:

1. **Define the problem**: what the model predicts and why it is a classification problem
2. **Set up** a Python virtual environment
3. **Build the dataset** from the Open-Meteo API: JSON → table → label → clean → `.parquet`

After that, `train_model.py` trains 4 algorithms on the dataset and prints the comparison board (part 4 at the end).

---

## 1. Problem definition

| | |
|---|---|
| **Question** | Given the weather **at this hour** in a city, **will it rain in the next hour?** |
| **Type** | Supervised learning → **binary classification** |
| **Input (features X)** | 8 hourly measurements: temperature, relative humidity, dew point, sea-level pressure, cloud cover, wind speed, wind direction, current precipitation |
| **Output (label y)** | `rain_next_hour` = `1` if precipitation in the next hour ≥ 0.1 mm, else `0` |
| **Data** | 5 Vietnamese cities × 10 years (2015–2024) × 24 hours ≈ **438,000 rows** |

### Why this problem fits a first model

- **It is classification, not regression.** "Predict tomorrow's temperature" outputs a number (regression). Here the output is one of two classes: rain / no rain.
- **The label comes free from the data.** You don't label anything by hand: the label for hour *t* is simply the precipitation at hour *t+1*.
- **It teaches the most important evaluation lesson.** Most hours are dry, so the classes are **imbalanced**. A model that always says "no rain" already gets a high accuracy and is still useless. You will see for yourself why **F1-score** matters more than accuracy here.
- **It is big enough.** About 438k rows, so the speed and size differences between algorithms show up clearly.

---

## 2. Environment setup (Windows / PowerShell)

Open a terminal in your project folder (`ml_project`):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install requests pandas pyarrow scikit-learn xgboost joblib
```

> If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
> You should see `(.venv)` at the start of your prompt. Activate the venv every time you open a new terminal.

| Library | Used for |
|---|---|
| `requests` | Calling the HTTP API |
| `pandas` | Tables (DataFrame), cleaning, saving files |
| `pyarrow` | Engine that lets pandas read and write `.parquet` |
| `scikit-learn` | Logistic Regression, Decision Tree, Random Forest, metrics, tuning |
| `xgboost` | XGBoost |
| `joblib` | Saving trained models to disk |

---

## 3. Getting the dataset from Open-Meteo

### 3.1 Understand the API first

Open-Meteo has a free **Historical Weather API** (no API key needed):

```
https://archive-api.open-meteo.com/v1/archive
```

Paste this URL into your browser. It asks for 1 day of temperature and precipitation in Hanoi:

```
https://archive-api.open-meteo.com/v1/archive?latitude=21.03&longitude=105.85&start_date=2024-01-01&end_date=2024-01-01&hourly=temperature_2m,precipitation&timezone=auto
```

| Parameter | Meaning |
|---|---|
| `latitude`, `longitude` | Location |
| `start_date`, `end_date` | Date range (`YYYY-MM-DD`) |
| `hourly` | Comma-separated list of variables you want, one value per hour |
| `timezone=auto` | Return times in the location's local timezone |

The response is **JSON** (shortened):

```json
{
  "latitude": 21.05,
  "longitude": 105.81,
  "timezone": "Asia/Bangkok",
  "hourly_units": { "time": "iso8601", "temperature_2m": "°C", "precipitation": "mm" },
  "hourly": {
    "time":           ["2024-01-01T00:00", "2024-01-01T01:00", "..."],
    "temperature_2m": [18.5, 18.1, "..."],
    "precipitation":  [0.0, 0.0, "..."]
  }
}
```

**Key observation:** `hourly` is a dictionary of **equal-length lists**, one list per variable, and position *i* in every list is the same hour. That is already column-shaped data, so `pd.DataFrame(data["hourly"])` turns it into a table in one line.

You can browse all available variables at https://open-meteo.com/en/docs/historical-weather-api.

### 3.2 The pipeline

```
API (JSON)  →  1. Fetch  →  2. JSON to table  →  3. Add label  →  4. Clean  →  5. Save .parquet
```

Each step is one function in the script below. What each one does:

#### Step 1: Fetch (`fetch_year`)

One request per **city per year** (5 cities × 10 years = 50 requests). Why not one request per city for all 10 years? In testing, the 10-year requests (~87k hours each) often hit the timeout, while one-year requests came back in about 2 seconds. Small requests are more reliable.

The free API also has a rate limit (requests covering many days count as several "calls"). So the script:

- waits 2 seconds between requests, and
- if a request fails (network error, timeout, or an HTTP code other than `200 OK`, e.g. `429 Too Many Requests`), waits 60 seconds and retries, up to 5 times.

#### Step 2: JSON → table (`json_to_table`)

```python
table = pd.DataFrame(data["hourly"])          # dict of lists -> columns
table["time"] = pd.to_datetime(table["time"]) # text -> real datetime
table.insert(0, "city", city)                 # remember which city each row is from
```

Then all 50 small tables (5 cities × 10 years) are stacked into one with `pd.concat`.

#### Step 3: Add the label (`add_label`)

```python
df = df.sort_values(["city", "time"])
df["next_precipitation"] = df.groupby("city")["precipitation"].shift(-1)
df["rain_next_hour"] = (df["next_precipitation"] >= 0.1).astype(int)
```

- `shift(-1)` moves the column up by one row, so each row sees the **next hour's** precipitation.
- `groupby("city")` keeps Hanoi's last hour from "seeing" Hai Phong's first hour.
- `>= 0.1` turns the number into a class: `1` = rain, `0` = no rain. 0.1 mm is the smallest amount usually reported as rain.

> **Why label *before* cleaning?** If cleaning deleted a row first, `shift(-1)` would pair an hour with the hour *two* steps later, and the label would be wrong. Building the label on the complete time series avoids that.

#### Step 4: Clean (`clean`)

| Check | Why |
|---|---|
| `drop_duplicates(["city", "time"])` | The same hour must not appear twice |
| `dropna()` | Removes missing values. It also removes the last hour of each city, which has no "next hour" and so no label |
| Humidity and cloud cover in 0–100, wind direction in 0–360 | Physically impossible values mean broken records |
| Precipitation ≥ 0, wind speed ≥ 0 | Same reason |
| **Drop `next_precipitation`** | **Very important.** This column *is* the answer. If you keep it, the model just reads the answer and scores ~100% (this is called **data leakage**) |

Open-Meteo data is already clean (no missing values in testing), so only **5 rows** are removed: the last hour of each city, which has no label. Doing these checks is still a habit worth building, because most real datasets are much messier.

#### Step 5: Save as Parquet

```python
df.to_parquet("weather_dataset.parquet", index=False)
```

Parquet vs CSV: Parquet is compressed (around 5× smaller here), loads faster, and **keeps the column types** (a datetime stays a datetime). CSV is plain text you can open in Excel. If you prefer CSV, use `df.to_csv("weather_dataset.csv", index=False)` and change `DATA_PATH` in `train_model.py`. The training script reads both.

### 3.3 Full script

Save this as `get_dataset.py` in your project folder (venv activated), then run `python get_dataset.py`. It takes about 3–5 minutes.

```python
import time

import pandas as pd
import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_YEAR = 2015
END_YEAR = 2024
RAIN_THRESHOLD_MM = 0.1
OUTPUT_FILE = "weather_dataset.parquet"

CITIES = {
    "Hanoi": (21.03, 105.85),
    "Hai Phong": (20.86, 106.68),
    "Da Nang": (16.05, 108.22),
    "Ho Chi Minh City": (10.82, 106.63),
    "Can Tho": (10.04, 105.79),
}

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]


def fetch_year(city, latitude, longitude, year):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(VARIABLES),
        "timezone": "auto",
    }
    for attempt in range(1, 6):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            if response.status_code == 200:
                return response.json()
            print(f"  {city} {year}: HTTP {response.status_code} (attempt {attempt})")
        except requests.RequestException as error:
            print(f"  {city} {year}: {error} (attempt {attempt})")
        time.sleep(60)
    raise RuntimeError(f"Could not download {city} {year}")


def json_to_table(data, city):
    table = pd.DataFrame(data["hourly"])
    table["time"] = pd.to_datetime(table["time"])
    table.insert(0, "city", city)
    return table


def add_label(df):
    df = df.sort_values(["city", "time"]).reset_index(drop=True)
    df["next_precipitation"] = df.groupby("city")["precipitation"].shift(-1)
    df["rain_next_hour"] = (df["next_precipitation"] >= RAIN_THRESHOLD_MM).astype(int)
    return df


def clean(df):
    rows_before = len(df)
    df = df.drop_duplicates(subset=["city", "time"])
    df = df.dropna()
    df = df[df["relative_humidity_2m"].between(0, 100)]
    df = df[df["cloud_cover"].between(0, 100)]
    df = df[df["wind_direction_10m"].between(0, 360)]
    df = df[(df["precipitation"] >= 0) & (df["wind_speed_10m"] >= 0)]
    df = df.drop(columns=["next_precipitation"])
    print(f"Cleaning removed {rows_before - len(df)} rows")
    return df.reset_index(drop=True)


def main():
    tables = []
    for city, (latitude, longitude) in CITIES.items():
        print(f"Downloading {city} ...")
        for year in range(START_YEAR, END_YEAR + 1):
            data = fetch_year(city, latitude, longitude, year)
            tables.append(json_to_table(data, city))
            time.sleep(2)

    df = pd.concat(tables, ignore_index=True)
    df = add_label(df)
    df = clean(df)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nSaved {OUTPUT_FILE}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.head())
    print("\nLabel distribution (0 = no rain, 1 = rain next hour):")
    print(df["rain_next_hour"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
```

### 3.4 Check the result

Expected output (the first-5-rows table printed by `df.head()` is left out here):

```
Downloading Hanoi ...
Downloading Hai Phong ...
Downloading Da Nang ...
Downloading Ho Chi Minh City ...
Downloading Can Tho ...
Cleaning removed 5 rows

Saved weather_dataset.parquet: 438355 rows x 11 columns

Label distribution (0 = no rain, 1 = rain next hour):
rain_next_hour
0    0.72
1    0.28
```

These numbers were measured from the real API (Nov 2026). Rain ratio per city: Hanoi 27.0%, Hai Phong 25.8%, Da Nang 33.0%, Ho Chi Minh City 26.9%, Can Tho 27.0%.

Things to notice:

- **438,355 rows**, which is well above 100k.
- **11 columns:** `city` and `time` (identifiers, not used as features), the 8 weather features, and the label `rain_next_hour`.
- **Label distribution:** only **~28%** of hours are followed by rain. That is the class imbalance you will deal with in training. A model that always says "no rain" gets **72% accuracy** without learning anything.

---

## 4. Next: train the models

```powershell
python train_model.py
```

It takes about 5–10 minutes depending on your CPU. Progress is printed for every algorithm. What the script does, in order:

| Step | What happens | Why |
|---|---|---|
| Load | Reads `weather_dataset.parquet` | |
| Split | **Train = 2015–2022, Test = 2023–2024** (split by time, not randomly) | Weather in hour *t* and hour *t+1* is almost the same. A random split puts neighbouring hours in both train and test, so the test score would look better than it really is. Splitting by time copies real use: learn from the past, predict the future. |
| Handle imbalance | `class_weight="balanced"` (sklearn), `scale_pos_weight` (XGBoost) | Makes mistakes on the rare "rain" class cost more, so the model doesn't just predict "no rain" every time |
| Scale | `StandardScaler` for Logistic Regression only | LR is sensitive to feature scale (pressure ~1000 vs cloud cover 0–100). Tree models split on thresholds, so they don't need scaling |
| Tune | `RandomizedSearchCV` + `TimeSeriesSplit(3)`, scoring = F1, on a 50k-row sample | Tries several hyperparameter combinations and keeps the best by cross-validated F1. The sample keeps tuning fast |
| Train | Refit the best settings on the **full** train set and time it | → `fit_time_s` |
| Evaluate | Predict on the test set | → accuracy, precision, recall, F1, ROC-AUC |
| Benchmark | Time `predict()` on 1000 rows, and save the model to measure its file size | → `predict_1000_ms`, `model_size_mb` |
| Decide | Prints the board and explains which algorithm wins and why | |

### How to read the board

| Metric | Meaning in this problem |
|---|---|
| `accuracy` | % of hours predicted correctly. **Misleading here:** "always no rain" already scores high (the baseline line under the board shows this) |
| `precision` | When the model says "rain", how often it is right |
| `recall` | Of all the hours that really were followed by rain, how many the model caught |
| `f1` | Harmonic mean of precision and recall. **Main metric**, because it is only high when both are high |
| `roc_auc` | How well the model *ranks* rainy hours above dry ones across all thresholds (0.5 = random, 1.0 = perfect) |
| `fit_time_s` | Training time on ~350k rows |
| `predict_1000_ms` | Inference speed. Matters if you serve predictions in an app |
| `model_size_mb` | Disk/memory cost. Random Forests with deep trees get large |

**Selection rule used by the script:** rank by F1. If several models are within 0.01 F1 of the best, they are equally good for practical purposes, so pick the **smallest and fastest** one. In real projects you rarely pick the single highest number. You pick the best trade-off between quality and cost.

Outputs are written to `outputs/`: one `.joblib` per algorithm, `best_model.joblib`, and `comparison_board.csv`.
