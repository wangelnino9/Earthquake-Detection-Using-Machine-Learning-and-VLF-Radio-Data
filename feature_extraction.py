import os
import re
import pandas as pd
import numpy as np
from datetime import datetime


DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIONS = ["AKT", "ANA", "IMZ", "KMK", "KTU", "NSB", "STU", "TYH"]

# Station coordinates (latitude, longitude)
STATION_COORDS = {
    "AKT": (39.75, 140.10),
    "ANA": (36.24, 137.98),
    "IMZ": (34.92, 136.98),
    "KMK": (43.81, 144.17),
    "KTU": (39.13, 141.49),
    "NSB": (43.57, 145.60),
    "STU": (33.57, 131.37),
    "TYH": (34.73, 138.98),
}

# 4-hour windows in seconds from midnight
WINDOWS = [
    (0,     14400,  "00:00-04:00"),   # Window 0 — nighttime end
    (14400, 28800,  "04:00-08:00"),   # Window 1 — early morning
    (28800, 43200,  "08:00-12:00"),   # Window 2 — morning
    (43200, 57600,  "12:00-16:00"),   # Window 3 — afternoon
    (57600, 72000,  "16:00-20:00"),   # Window 4 — evening
    (72000, 86400,  "20:00-24:00"),   # Window 5 — nighttime start
]


def extract_date(filename):
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(
                match.group(1), "%Y%m%d").date()
        except:
            return None
    return None

def is_bad_file(filename):
    return ("MISS" in filename.upper() or
            "LACK" in filename.upper())


def read_vlf_file(filepath):
    try:
        df = pd.read_csv(
            filepath,
            comment="%",
            sep=r"\s+",
            header=None,
            names=["time", "amplitude", "phase"]
        )
        df = df.dropna()
        df = df[(df["time"] >= 0) & (df["time"] <= 86400)]
        return df
    except Exception:
        return None

def extract_window_features(df, date, station):
    lat, lon = STATION_COORDS[station]
    rows = []

    for win_idx, (t_start, t_end, win_label) in enumerate(WINDOWS):
        # Filter rows within this time window
        mask   = (df["time"] >= t_start) & (df["time"] < t_end)
        window = df[mask]

        # Skip if too few readings
        if len(window) < 7200:
            continue

        amp   = window["amplitude"]
        phase = window["phase"]

        rows.append({
            "date":       date,
            "station":    station,
            "latitude":   lat,
            "longitude":  lon,
            "window_id":  win_idx,
            "window":     win_label,
            "Amp_Mean":   round(float(amp.mean()),   4),
            "Amp_Std":    round(float(amp.std()),    4),
            "Phase_Mean": round(float(phase.mean()), 4),
            "Phase_Std":  round(float(phase.std()),  4),
        })

    return rows

def extract_all():

    all_records = []
    skipped     = []

    for station in STATIONS:
        station_path = os.path.join(DATA_DIR, station, "JJI")

        if not os.path.exists(station_path):
            print(f"  Folder not found: {station_path}")
            continue

        files     = sorted(os.listdir(station_path))
        txt_files = [f for f in files if f.endswith(".txt")]

        print(f"\nProcessing {station} — {len(txt_files)} files...")

        station_count   = 0
        station_skipped = 0

        for filename in txt_files:

            # Skip MISS and LACK files
            if is_bad_file(filename):
                date = extract_date(filename)
                skipped.append({
                    "station":  station,
                    "filename": filename,
                    "date":     date,
                    "reason":   "MISS/LACK"
                })
                station_skipped += 1
                continue

            # Extract date
            date = extract_date(filename)
            if date is None:
                skipped.append({
                    "station":  station,
                    "filename": filename,
                    "date":     None,
                    "reason":   "date_parse_failed"
                })
                station_skipped += 1
                continue
            filepath = os.path.join(station_path, filename)
            df       = read_vlf_file(filepath)

            if df is None or len(df) < 1000:
                skipped.append({
                    "station":  station,
                    "filename": filename,
                    "date":     date,
                    "reason":   "too_few_rows"
                })
                station_skipped += 1
                continue

            # Extract 4-hour window features
            records = extract_window_features(df, date, station)

            if not records:
                skipped.append({
                    "station":  station,
                    "filename": filename,
                    "date":     date,
                    "reason":   "no_valid_windows"
                })
                station_skipped += 1
                continue

            all_records.extend(records)
            station_count += 1

        print(f"  Extracted : {station_count} files "
              f"({station_count * len(WINDOWS)} rows)")
        print(f"  Skipped   : {station_skipped} files")

    #Build DataFrame
    df_features = pd.DataFrame(all_records)
    df_features = df_features.sort_values(
        ["station", "date", "window_id"]
    ).reset_index(drop=True)

    #Save features
    output_path = os.path.join(OUTPUT_DIR, "vlf_features.csv")
    df_features.to_csv(output_path, index=False)

    print(f"\n── Features saved to {output_path}")
    print(f"── Total rows    : {len(df_features):,}")
    print(f"── Columns       : {list(df_features.columns)}")
    print(f"\nFirst 5 rows:")
    print(df_features.head())
    print(f"\nBasic stats:")
    print(df_features[["Amp_Mean", "Amp_Std",
                        "Phase_Mean", "Phase_Std"]].describe())

    #Save skipped files log
    df_skipped    = pd.DataFrame(skipped)
    skipped_path  = os.path.join(OUTPUT_DIR, "skipped_files.csv")
    df_skipped.to_csv(skipped_path, index=False)
    print(f"\n── Skipped files log saved to {skipped_path}")
    print(f"── Total skipped : {len(df_skipped)}")

    return df_features

if __name__ == "__main__":
    print("=" * 55)
    print("SFEATURE EXTRACTION (4-HOUR WINDOWS)")
    print("=" * 55)
    print("  6 windows × 4 features = 24 features per day")
    print("  Rows per file : up to 6 ")
    print()
    df = extract_all()