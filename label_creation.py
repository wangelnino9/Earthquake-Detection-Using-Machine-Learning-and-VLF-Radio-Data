"""
Step 4 — Label Creation
Merges VLF feature dataset (Step 2) with earthquake days (Step 3)
to create a labelled ML-ready dataset.

    label = 1 only if the station's own path matched an earthquake
    label = 0 if no earthquake fell in that station's Fresnel squares

Input files:
    outputs/vlf_features.csv     — from Step 2
    outputs/earthquake_days.csv  — from Step 3

Output files:
    outputs/vlf_labelled.csv     — full labelled dataset
    
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load Input Files

def load_inputs():
    """Load VLF features and earthquake days from Step 2 and Step 3."""

    print("Loading input files...")

    # Load VLF features
    features_path = os.path.join(OUTPUT_DIR, "vlf_features.csv")
    if not os.path.exists(features_path):
        print(f"  ERROR: {features_path} not found.")
        print("  Please run Step 2 first.")
        return None, None

    df_features = pd.read_csv(features_path)
    df_features["date"] = pd.to_datetime(df_features["date"]).dt.date
    print(f"  VLF features loaded   : {len(df_features):,} rows")
    print(f"  Stations              : {sorted(df_features['station'].unique())}")
    print(f"  Date range            : {df_features['date'].min()} "
          f"to {df_features['date'].max()}")

    # Load earthquake days
    eq_path = os.path.join(OUTPUT_DIR, "earthquake_days.csv")
    if not os.path.exists(eq_path):
        print(f"  ERROR: {eq_path} not found.")
        print("  Please run Step 3 first.")
        return None, None

    df_eq = pd.read_csv(eq_path)
    df_eq["date"] = pd.to_datetime(df_eq["date"]).dt.date
    print(f"  Earthquake days loaded: {len(df_eq):,} unique dates")

    return df_features, df_eq


#Build Date-Station Lookup

def build_earthquake_lookup(df_eq):
    
    print("\nBuilding date-station earthquake lookup...")

    lookup = {}  # key: (date, station), value: 1

    for _, row in df_eq.iterrows():
        date = row["date"]
        paths = str(row["matched_paths"])

        # Split pipe-separated station names
        # e.g. "AKT|KTU|IMZ" → ["AKT", "KTU", "IMZ"]
        if paths and paths != "nan":
            stations_matched = [s.strip() for s in paths.split("|")
                                if s.strip()]
            for stn in stations_matched:
                lookup[(date, stn)] = 1

    print(f"  Earthquake (date, station) pairs: {len(lookup):,}")
    return lookup


#Assign Labels

def assign_labels(df_features, lookup):
    print("\nAssigning labels...")

    labels = []
    for _, row in df_features.iterrows():
        key = (row["date"], row["station"])
        labels.append(lookup.get(key, 0))

    df_features = df_features.copy()
    df_features["label"] = labels

    # Count labels
    n_eq     = sum(labels)
    n_no_eq  = len(labels) - n_eq
    pct_eq   = n_eq / len(labels) * 100

    print(f"  Total rows            : {len(df_features):,}")
    print(f"  Label = 1 (EQ day)    : {n_eq:,} ({pct_eq:.1f}%)")
    print(f"  Label = 0 (no EQ)     : {n_no_eq:,} ({100-pct_eq:.1f}%)")
    print(f"  Class imbalance ratio : 1 : {n_no_eq/n_eq:.1f}")

    return df_features


# Data Checks

def sanity_checks(df):
    
    print("\nRunning data checks...")
    passed = True

    # Check 1: No missing values in features
    feature_cols = ["Amp_Mean", "Amp_Std", "Night_Amp_Mean",
                    "Night_Amp_Std", "Phase_Mean", "Phase_Std"]
    missing = df[feature_cols].isnull().sum().sum()
    if missing > 0:
        print(f"  WARNING: {missing} missing values in feature columns")
        passed = False
    else:
        print(f" No missing values in feature columns")

    # Check 2: Labels are only 0 or 1
    unique_labels = df["label"].unique()
    if set(unique_labels) <= {0, 1}:
        print(f" Labels are correctly 0 and 1 only")
    else:
        print(f"  WARNING: Unexpected label values: {unique_labels}")
        passed = False

    # Check 3: All 8 stations present
    expected_stations = {"AKT", "ANA", "IMZ", "KMK",
                         "KTU", "NSB", "STU", "TYH"}
    actual_stations   = set(df["station"].unique())
    missing_stations  = expected_stations - actual_stations
    if missing_stations:
        print(f"  WARNING: Missing stations: {missing_stations}")
        passed = False
    else:
        print(f"All 8 stations present")

    # Check 4: Label distribution per station
    print(f"\n  Label distribution per station:")
    print(f"  {'Station':<8} {'Total':>8} {'EQ days':>10} "
          f"{'%':>8}")
    print("  " + "-"*38)
    for stn in sorted(df["station"].unique()):
        stn_df   = df[df["station"] == stn]
        total    = len(stn_df)
        eq_days  = stn_df["label"].sum()
        pct      = eq_days / total * 100
        print(f"  {stn:<8} {total:>8,} {eq_days:>10,} {pct:>7.1f}%")

    # Check 5: Date coverage
    print(f"\n  Date coverage per year:")
    df["year"] = pd.to_datetime(df["date"].astype(str)).dt.year
    for year, grp in df.groupby("year"):
        days     = grp["date"].nunique()
        eq_days  = grp[grp["label"] == 1]["date"].nunique()
        print(f"    {year}: {days:,} days total, "
              f"{eq_days:,} earthquake days")
    df = df.drop(columns=["year"])

    if passed:
        print("\n All data checks passed")
    else:
        print("\n Some checks flagged review warnings above")

    return df


#Visaulise

# def visualise(df):
    
    print("\nGenerating visualisation...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    colors = {
        0: "#5B8DB8",   # blue — no earthquake
        1: "#C0392B",   # red  — earthquake day
    }

    # ── Plot 1: Overall class distribution ───────────────────
    ax = axes[0, 0]
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(["No Earthquake\n(Label = 0)",
                   "Earthquake Day\n(Label = 1)"],
                  counts.values,
                  color=[colors[0], colors[1]],
                  width=0.5, edgecolor="white")

    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + len(df)*0.005,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="#2D3748")

    ax.set_title("Overall Class Distribution",
                 fontsize=12, fontweight="bold", color="#1B3A5C")
    ax.set_ylabel("Number of rows", fontsize=10)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # ── Plot 2: Label distribution per station ────────────────
    ax = axes[0, 1]
    stations  = sorted(df["station"].unique())
    eq_pcts   = []
    for stn in stations:
        stn_df = df[df["station"] == stn]
        eq_pcts.append(stn_df["label"].mean() * 100)

    bars = ax.bar(stations, eq_pcts,
                  color="#2E6DA4", edgecolor="white", width=0.6)
    for bar, pct in zip(bars, eq_pcts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{pct:.1f}%",
                ha="center", va="bottom",
                fontsize=9, color="#2D3748")

    ax.axhline(y=sum(eq_pcts)/len(eq_pcts),
               color="#C0392B", linewidth=1.2,
               linestyle="--", label="Mean %")
    ax.set_title("Earthquake Day % per Station",
                 fontsize=12, fontweight="bold", color="#1B3A5C")
    ax.set_ylabel("% earthquake days", fontsize=10)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)

    # ── Plot 3: Earthquake days over time ─────────────────────
    ax = axes[1, 0]
    df["month"] = pd.to_datetime(
        df["date"].astype(str)).dt.to_period("M")
    monthly = (df[df["label"] == 1]
               .groupby("month")["date"]
               .nunique()
               .reset_index())
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    ax.bar(monthly["month_dt"], monthly["date"],
           color="#C0392B", alpha=0.7,
           width=20, edgecolor="white")
    ax.set_title("Earthquake Days per Month",
                 fontsize=12, fontweight="bold", color="#1B3A5C")
    ax.set_ylabel("Number of earthquake days", fontsize=10)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    df = df.drop(columns=["month"])

    # ── Plot 4: Rows per station ──────────────────────────────
    ax = axes[1, 1]
    stn_counts = df.groupby("station")["label"].value_counts().unstack(fill_value=0)
    stn_counts = stn_counts.reindex(sorted(stn_counts.index))

    x = np.arange(len(stn_counts))
    w = 0.35
    ax.bar(x - w/2, stn_counts[0], width=w,
           color=colors[0], label="No EQ (0)", edgecolor="white")
    ax.bar(x + w/2, stn_counts[1], width=w,
           color=colors[1], label="EQ day (1)", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(stn_counts.index, fontsize=9)
    ax.set_title("Label Counts per Station",
                 fontsize=12, fontweight="bold", color="#1B3A5C")
    ax.set_ylabel("Number of rows", fontsize=10)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    plt.suptitle(
        "Step 4 — Label Creation Summary\n"
        "Option B: Path-level labelling",
        fontsize=13, fontweight="bold",
        color="#1B3A5C", y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "step4_label_summary.png")
    plt.savefig(out_path, dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved: {out_path}")


#Main

def main():
    print("=" * 55)
    print("STEP 4 — LABEL CREATION (OPTION B: PATH LEVEL)")
    print("=" * 55)

    #Load inputs
    df_features, df_eq = load_inputs()
    if df_features is None:
        return

    #Build lookup
    lookup = build_earthquake_lookup(df_eq)

    # Assign labels
    df_labelled = assign_labels(df_features, lookup)

    #Data checks
    df_labelled = sanity_checks(df_labelled)

    #Visualise
    # visualise(df_labelled)

    #Save output
    out_path = os.path.join(OUTPUT_DIR, "vlf_labelled.csv")
    df_labelled.to_csv(out_path, index=False)
    print(f"\n  Labelled dataset saved: {out_path}")
    print(f"  Shape: {df_labelled.shape[0]:,} rows × "
          f"{df_labelled.shape[1]} columns")
    print(f"  Columns: {list(df_labelled.columns)}")

    print("\nComplete")
    


if __name__ == "__main__":
    main()