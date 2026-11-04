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
