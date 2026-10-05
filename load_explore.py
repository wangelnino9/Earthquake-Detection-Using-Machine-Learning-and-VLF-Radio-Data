
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv() 
DATA_DIR = os.getenv("DATA_DIR")
# Configuration 
STATIONS = ["AKT", "ANA", "IMZ", "KMK", "KTU", "NSB", "STU", "TYH"]

#Read a single VLF .txt file 
def read_vlf_file(filepath):
    try:
        df = pd.read_csv(
            filepath,
            comment="%",           
            sep=r"\s+",            
            header=None,
            names=["time", "amplitude", "phase"]
        )
        return df
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None


#Explore a single file
def explore_single_file(station="IMZ"):
    station_path = os.path.join(DATA_DIR, station, "JJI")
    files = sorted(os.listdir(station_path))
    txt_files = [f for f in files if f.endswith(".txt")]

    if not txt_files:
        print(f"No .txt files found in {station_path}")
        return

    sample_file = os.path.join(station_path, txt_files[0])
    print(f"\n── Sample file: {sample_file}")

    df = read_vlf_file(sample_file)

    if df is not None:
        print(f"  Rows      : {len(df):,}")
        print(f"  Columns   : {list(df.columns)}")
        print(f"  Time range: {df['time'].min()} → {df['time'].max()}")
        print(f"\n  First 5 rows:")
        print(df.head())
        print(f"\n  Basic stats:")
        print(df.describe())

        # plot amplitude and phase for the day
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(df["time"], df["amplitude"], linewidth=0.5, color="steelblue")
        ax1.set_ylabel("Amplitude (dB)")
        ax1.set_title(f"Sample VLF file — {txt_files[0]}")
        ax2.plot(df["time"], df["phase"], linewidth=0.5, color="coral")
        ax2.set_ylabel("Phase (degrees)")
        ax2.set_xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig("outputs/sample_vlf_plot.png", dpi=150)
        plt.show()
        print("\n  Plot saved to outputs/sample_vlf_plot.png")


# Count all files across all stations
def count_all_files():
    """
    Count how many .txt files exist per station
    and print a summary.
    """
    print("\n── File count per station:")
    total = 0
    for station in STATIONS:
        station_path = os.path.join(DATA_DIR, station, "JJI")
        if not os.path.exists(station_path):
            print(f"  {station}: folder not found")
            continue
        files = [f for f in os.listdir(station_path) if f.endswith(".txt")]
        print(f"  {station}: {len(files)} files")
        total += len(files)
    print(f"\n  Total files across all stations: {total}")


# Check date coverage
def check_date_coverage():
    print("\n── Date coverage per station:")
    for station in STATIONS:
        station_path = os.path.join(DATA_DIR, station, "JJI")
        if not os.path.exists(station_path):
            continue
        files = sorted([f for f in os.listdir(station_path) if f.endswith(".txt")])
        if not files:
            continue

        # extract dates from filenames
        dates = []
        for f in files:
            try:
                date_str = f.replace(".txt", "")[-8:]  # last 8 chars = YYYYMMDD
                dates.append(pd.to_datetime(date_str, format="%Y%m%d"))
            except:
                pass

        if dates:
            print(f"  {station}: {min(dates).date()} → {max(dates).date()} "
                  f"({len(dates)} files)")


# Main
if __name__ == "__main__":

    # check output folder
    os.makedirs("outputs", exist_ok=True)

    # explore a single file
    explore_single_file(station="IMZ")

    # count all files
    count_all_files()

    # check date coverage
    check_date_coverage()