# Stage 2 : Feature Extraction from Raw VLF Files


import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

#Configuration 
DATA_DIR = "data"
STATIONS = ["AKT", "ANA", "IMZ", "KMK", "KTU", "NSB", "STU", "TYH"]
OUTPUT_DIR = "outputs"

# Nighttime window: 21:00 to 04:00 local time
# In seconds from midnight:
# 21:00 = 75600 seconds
# 04:00 = 14400 seconds
NIGHT_START = 75600
NIGHT_END   = 14400

# Station coordinates (latitude, longitude)
# These are approximate locations for each station
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

# Function: Extract date from filename 
def extract_date(filename):
    """
    Extract YYYYMMDD from filename using regex.
    Works for: JJI_STX20140101.txt
               JJI_STX20140102MISS.txt
               JJI_STX20140106LACK.txt
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").date()
        except:
            return None
    return None

# ── Function: Check if file is bad quality ───────────────────
def is_bad_file(filename):
    """Returns True if file is MISS or LACK."""
    return "MISS" in filename.upper() or "LACK" in filename.upper()

# ── Function: Read a single VLF file ─────────────────────────
def read_vlf_file(filepath):
    """Read a VLF .txt file, skip header lines starting with %."""
    try:
        df = pd.read_csv(
            filepath,
            comment="%",
            sep=r"\s+",
            header=None,
            names=["time", "amplitude", "phase"]
        )
        # drop any rows with NaN
        df = df.dropna()
        # keep only valid rows (time between 0 and 86400)
        df = df[(df["time"] >= 0) & (df["time"] <= 86400)]
        return df
    except Exception as e:
        return None

# Function: Extract features from one day's data 
def extract_features(df):
    """
    Given a DataFrame for one day, compute daily features.
    Returns a dict of features.
    """
    amp = df["amplitude"]
    phase = df["phase"]

    # nighttime mask: 21:00 onwards OR before 04:00
    night_mask = (df["time"] >= NIGHT_START) | (df["time"] <= NIGHT_END)
    night_amp = df.loc[night_mask, "amplitude"]

    features = {
        "Amp_Mean":        round(amp.mean(), 4),
        "Amp_Std":         round(amp.std(), 4),
        "Phase_Mean":      round(phase.mean(), 4),
        "Phase_Std":       round(phase.std(), 4),
        "Night_Amp_Mean":  round(night_amp.mean(), 4) if len(night_amp) > 0 else np.nan,
        "Night_Amp_Std":   round(night_amp.std(), 4)  if len(night_amp) > 0 else np.nan,
    }
    return features

#  Main: Process all stations and all files 
def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []
    skipped = []

    for station in STATIONS:
        station_path = os.path.join(DATA_DIR, station, "JJI")

        if not os.path.exists(station_path):
            print(f"  Folder not found: {station_path}")
            continue

        files = sorted(os.listdir(station_path))
        txt_files = [f for f in files if f.endswith(".txt")]

        print(f"\nProcessing {station} — {len(txt_files)} files...")

        station_count = 0
        station_skipped = 0

        for filename in txt_files:

            # skip MISS and LACK files
            if is_bad_file(filename):
                date = extract_date(filename)
                skipped.append({"station": station,
                                 "filename": filename,
                                 "date": date,
                                 "reason": "MISS/LACK"})
                station_skipped += 1
                continue

            # extract date
            date = extract_date(filename)
            if date is None:
                skipped.append({"station": station,
                                 "filename": filename,
                                 "date": None,
                                 "reason": "date_parse_failed"})
                station_skipped += 1
                continue

            # read file
            filepath = os.path.join(station_path, filename)
            df = read_vlf_file(filepath)

            if df is None or len(df) < 1000:
                skipped.append({"station": station,
                                 "filename": filename,
                                 "date": date,
                                 "reason": "too_few_rows"})
                station_skipped += 1
                continue

            # extract features
            features = extract_features(df)

            # build record
            lat, lon = STATION_COORDS[station]
            record = {
                "date":     date,
                "station":  station,
                "latitude": lat,
                "longitude": lon,
                **features
            }
            all_records.append(record)
            station_count += 1

        print(f"  Extracted : {station_count} records")
        print(f"  Skipped   : {station_skipped} files")

    # build final DataFrame
    df_features = pd.DataFrame(all_records)
    df_features = df_features.sort_values(["station", "date"]).reset_index(drop=True)

    # save to CSV
    output_path = os.path.join(OUTPUT_DIR, "vlf_features.csv")
    df_features.to_csv(output_path, index=False)
    print(f"\n── Features saved to {output_path}")
    print(f"── Total records: {len(df_features)}")
    print(f"\nFirst 5 rows:")
    print(df_features.head())
    print(f"\nColumn list:")
    print(list(df_features.columns))
    print(f"\nBasic stats:")
    print(df_features.describe())

    # save skipped files log
    df_skipped = pd.DataFrame(skipped)
    skipped_path = os.path.join(OUTPUT_DIR, "skipped_files.csv")
    df_skipped.to_csv(skipped_path, index=False)
    print(f"\n── Skipped files log saved to {skipped_path}")
    print(f"── Total skipped: {len(df_skipped)}")

    return df_features

# Run 
if __name__ == "__main__":
    df = extract_all()
    print("\n── Step 2 complete.")