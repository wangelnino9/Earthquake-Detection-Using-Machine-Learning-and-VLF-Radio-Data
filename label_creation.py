import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import timedelta

OUTPUT_DIR      = "outputs"
PRE_WINDOW_DAYS = 6
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = ["Amp_Mean", "Amp_Std", "Phase_Mean", "Phase_Std"]


def load_inputs():
    print("Loading input files...")
    features_path = os.path.join(OUTPUT_DIR, "vlf_features.csv")
    if not os.path.exists(features_path):
        print(f"  ERROR: {features_path} not found.")
        return None, None
    df_features = pd.read_csv(features_path)
    df_features["date"] = pd.to_datetime(df_features["date"]).dt.date
    print(f"  VLF features loaded   : {len(df_features):,} rows")
    print(f"  Stations              : {sorted(df_features['station'].unique())}")
    print(f"  Date range            : {df_features['date'].min()} to {df_features['date'].max()}")

    eq_path = os.path.join(OUTPUT_DIR, "earthquake_days.csv")
    if not os.path.exists(eq_path):
        print(f"  ERROR: {eq_path} not found.")
        return None, None
    df_eq = pd.read_csv(eq_path)
    df_eq["date"] = pd.to_datetime(df_eq["date"]).dt.date
    print(f"  Earthquake days loaded: {len(df_eq):,} unique dates")
    return df_features, df_eq


def build_pre_window_lookup(df_eq):
    print(f"\nBuilding pre-earthquake window lookup...")
    print(f"  Pre-earthquake window : {PRE_WINDOW_DAYS} days")
    print(f"  Labelling             : Path level (Option B)")

    lookup = {}
    skipped = 0

    for _, row in df_eq.iterrows():
        eq_date = row["date"]
        paths   = str(row["matched_paths"])

        if not paths or paths == "nan" or paths.strip() == "":
            skipped += 1
            continue

        # Get affected stations
        stations_affected = [
            s.strip() for s in paths.replace("|", ",").split(",")
            if s.strip()
        ]
        if not stations_affected:
            skipped += 1
            continue

        # Label earthquake day AND 7 days before
        for days_before in range(0, PRE_WINDOW_DAYS + 1):
            pre_date = eq_date - timedelta(days=days_before)
            for stn in stations_affected:
                lookup[(pre_date, stn)] = 1

    print(f"  Earthquake events     : {len(df_eq):,}")
    print(f"  Skipped (no path)     : {skipped:,}")
    print(f"  Unique (date,stn)     : {len(lookup):,}")
    return lookup

def assign_labels(df_features, lookup):
    print("\nMarking pre-seismic periods...")
    df_features = df_features.copy()
    df_features["label"] = df_features.apply(
        lambda row: lookup.get(
            (row["date"], row["station"]), 0),
        axis=1
    )
    n_total = len(df_features)
    n_eq    = df_features["label"].sum()
    print(f"  Total rows         : {n_total:,}")
    print(f"  Pre-seismic rows   : {n_eq:,} ({n_eq/n_total*100:.1f}%)")
    print(f"  Normal rows        : {n_total-n_eq:,}")
    return df_features

def sanity_checks(df):
    print("\nRunning sanity checks...")
    missing = df[FEATURE_COLS].isnull().sum().sum()
    print(f"  {'✓' if missing == 0 else 'WARNING'} Missing values: {missing}")
    print(f"  ✓ Labels: {sorted(df['label'].unique())}")
    print(f"  ✓ Stations: {sorted(df['station'].unique())}")

    print(f"\n  Label % per station:")
    print(f"  {'Station':<8} {'Total':>8} {'EQ rows':>10} {'%':>8}")
    print("  " + "-"*38)
    for stn in sorted(df["station"].unique()):
        stn_df = df[df["station"] == stn]
        eq_rows = stn_df["label"].sum()
        pct = eq_rows / len(stn_df) * 100
        print(f"  {stn:<8} {len(stn_df):>8,} {eq_rows:>10,} {pct:>7.1f}%")

    if "window_id" in df.columns:
        print(f"\n  Label % per window:")
        print(f"  {'Window':<18} {'Total':>8} {'EQ rows':>10} {'%':>8}")
        print("  " + "-"*46)
        for win_id in sorted(df["window_id"].unique()):
            win_df  = df[df["window_id"] == win_id]
            win_lbl = win_df["window"].iloc[0] if "window" in df.columns else str(win_id)
            eq_rows = win_df["label"].sum()
            pct = eq_rows / len(win_df) * 100
            print(f"  {win_lbl:<18} {len(win_df):>8,} {eq_rows:>10,} {pct:>7.1f}%")

    print(f"\n  Date coverage per year:")
    df["year"] = pd.to_datetime(df["date"].astype(str)).dt.year
    for year, grp in df.groupby("year"):
        days    = grp["date"].nunique()
        eq_days = grp[grp["label"] == 1]["date"].nunique()
        print(f"    {year}: {days:,} total days, {eq_days:,} labelled ({eq_days/days*100:.1f}%)")
    df = df.drop(columns=["year"])
    print("  ✓ Sanity checks complete")
    return df

def assign_eq_only_labels(df_features, df_eq):
    print("\nCreating earthquake-day-only labels...")

    eq_lookup = {}
    for _, row in df_eq.iterrows():
        eq_date = row["date"]
        paths   = str(row["matched_paths"])
        if not paths or paths == "nan":
            continue
        stations = [s.strip() for s in
                    paths.replace("|", ",").split(",")
                    if s.strip()]
        for stn in stations:
            eq_lookup[(eq_date, stn)] = 1

    df_features = df_features.copy()
    df_features["eq_label"] = df_features.apply(
        lambda row: eq_lookup.get(
            (row["date"], row["station"]), 0),
        axis=1
    )

    n_eq = df_features["eq_label"].sum()
    print(f"  Earthquake day rows : {n_eq:,} "
          f"({n_eq/len(df_features)*100:.1f}%)")
    print(f"  Normal rows         : {len(df_features)-n_eq:,}")

    return df_features

def visualise(df):
    print("\nGenerating visualisation...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    col_eq = "#C0392B"; col_no = "#2E6DA4"

    ax = axes[0]
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(["Normal\n(0)", f"Pre-EQ\n(1)"],
                  counts.values, color=[col_no, col_eq],
                  width=0.5, edgecolor="white")
    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + len(df)*0.005,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#000000")
    ax.set_title(f"Class Distribution\nPath Level + {PRE_WINDOW_DAYS}-Day Window",
                 fontsize=11, fontweight="bold", color="#1B3A5C")
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    stations = sorted(df["station"].unique())
    eq_pcts  = [df[df["station"]==s]["label"].mean()*100 for s in stations]
    bars = ax.bar(stations, eq_pcts, color=col_eq, alpha=0.8,
                  edgecolor="white", width=0.6)
    for bar, pct in zip(bars, eq_pcts):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.3, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9, color="#000000")
    ax.axhline(df["label"].mean()*100, color="#1B3A5C",
               linewidth=1.2, linestyle="--", label="Overall mean")
    ax.set_title("Pre-EQ % per Station\n(path level — varies by station)",
                 fontsize=11, fontweight="bold", color="#1B3A5C")
    ax.set_ylabel("% pre-EQ rows"); ax.legend(fontsize=9)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    df["month"] = pd.to_datetime(df["date"].astype(str)).dt.to_period("M")
    monthly = (df[df["label"]==1].groupby("month")["date"]
               .nunique().reset_index())
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    ax.bar(monthly["month_dt"], monthly["date"],
           color=col_eq, alpha=0.7, width=20, edgecolor="white")
    ax.set_title(f"Labelled Days per Month\n(EQ + {PRE_WINDOW_DAYS} days before)",
                 fontsize=11, fontweight="bold", color="#1B3A5C")
    ax.set_ylabel("Number of labelled days")
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    df = df.drop(columns=["month"])

    plt.suptitle(
        f"Path Level + {PRE_WINDOW_DAYS}-Day Pre-Earthquake Window\n"
        "4-hour windows — all windows of labelled days = 1",
        fontsize=12, fontweight="bold", color="#1B3A5C", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "label_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved: {out_path}")


def main():
    print("=" * 60)
    print("LABEL CREATION")
    print(f"  Approach : Path Level (Option B)")
    print(f"  Window   : {PRE_WINDOW_DAYS}-day pre-earthquake")
    print( "  Rows     : All 6 four-hour windows labelled")
    print("=" * 60)

    df_features, df_eq = load_inputs()
    if df_features is None:
        return

    lookup      = build_pre_window_lookup(df_eq)
    df_labelled = assign_labels(df_features, lookup)
    df_labelled = sanity_checks(df_labelled)
    visualise(df_labelled)
#create earthquake-day-only labels for CNN/LSTM
    df_eq_only = assign_eq_only_labels(df_labelled, df_eq)
    eq_only_path = os.path.join(OUTPUT_DIR, "vlf_eq_only_labels.csv")
    df_eq_only[["date", "station", "window_id",
                "window"] + FEATURE_COLS + ["eq_label"]].to_csv(
        eq_only_path, index=False)
    print(f"  EQ-only labels saved : {eq_only_path}")
    out_path = os.path.join(OUTPUT_DIR, "vlf_labelled.csv")
    df_labelled.to_csv(out_path, index=False)
    print(f"\n  Labelled dataset saved : {out_path}")
    print(f"  Shape  : {df_labelled.shape[0]:,} rows × "
          f"{df_labelled.shape[1]} columns")
    print(f"  Columns: {list(df_labelled.columns)}")
    print(f"\ncomplete — path level + {PRE_WINDOW_DAYS}-day window")


if __name__ == "__main__":
    main()