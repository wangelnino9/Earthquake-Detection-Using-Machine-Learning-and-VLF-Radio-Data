

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature columns to preprocess
FEATURE_COLS = [
    "Amp_Mean",
    "Amp_Std",
    "Night_Amp_Mean",
    "Night_Amp_Std",
    "Phase_Mean",
    "Phase_Std",
]

# Percentile bounds for capping
LOWER_PCT = 1   # cap values below 1st percentile
UPPER_PCT = 99  # cap values above 99th percentile


#Load Data 

def load_data():
    """Load labelled dataset from last stage"""
    print("Loading labelled dataset...")

    path = os.path.join(OUTPUT_DIR, "vlf_labelled.csv")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        print("  Please run stage 4 first.")
        return None

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    print(f"  Rows loaded    : {len(df):,}")
    print(f"  Columns        : {list(df.columns)}")
    print(f"  Label = 1 (EQ) : {df['label'].sum():,} "
          f"({df['label'].mean()*100:.1f}%)")
    print(f"  Label = 0      : {(df['label']==0).sum():,} "
          f"({(1-df['label'].mean())*100:.1f}%)")
    return df


#Check Missing Values 

def check_missing(df):
    """Check and report any missing values."""
    print("\nChecking for missing values...")

    missing = df[FEATURE_COLS].isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        print("No missing values found in any feature column")
    else:
        print(f"  WARNING: {total_missing} missing values found:")
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count} missing")
        print("  Filling missing values with station median...")
        for col in FEATURE_COLS:
            if df[col].isnull().any():
                df[col] = df.groupby("station")[col].transform(
                    lambda x: x.fillna(x.median())
                )
        print("Missing values filled")

    return df


# Percentile Capping 

def percentile_capping(df):
    
    print(f"\nApplying percentile capping "
          f"({LOWER_PCT}th – {UPPER_PCT}th percentile)...")
    print("  Capping per feature per station\n")

    df_capped = df.copy()
    caps_record = []

    stations = sorted(df["station"].unique())

    print(f"  {'Station':<8} {'Feature':<18} "
          f"{'Lower cap':>12} {'Upper cap':>12} "
          f"{'Values capped':>15}")
    print("  " + "-" * 68)

    for stn in stations:
        stn_mask = df_capped["station"] == stn

        for col in FEATURE_COLS:
            # Get values for this station
            values = df_capped.loc[stn_mask, col]

            # Calculate percentile caps
            lower_cap = np.percentile(values, LOWER_PCT)
            upper_cap = np.percentile(values, UPPER_PCT)

            # Count how many values will be capped
            n_lower = (values < lower_cap).sum()
            n_upper = (values > upper_cap).sum()
            n_capped = n_lower + n_upper

            # Apply capping — clip replaces values outside bounds
            df_capped.loc[stn_mask, col] = values.clip(
                lower=lower_cap,
                upper=upper_cap
            )

            # Record the caps for documentation
            caps_record.append({
                "station":    stn,
                "feature":    col,
                "lower_cap":  round(lower_cap, 4),
                "upper_cap":  round(upper_cap, 4),
                "n_capped":   n_capped,
                "pct_capped": round(n_capped / len(values) * 100, 2)
            })

            print(f"  {stn:<8} {col:<18} "
                  f"{lower_cap:>12.4f} {upper_cap:>12.4f} "
                  f"{n_capped:>12} ({n_capped/len(values)*100:.1f}%)")

    caps_df = pd.DataFrame(caps_record)

    # Summary
    total_capped = caps_df["n_capped"].sum()
    total_values = len(df) * len(FEATURE_COLS)
    print(f"\n  Total values capped : {total_capped:,} "
          f"/ {total_values:,} "
          f"({total_capped/total_values*100:.2f}%)")

    return df_capped, caps_df


# Verify Preprocessing 

def verify_preprocessing(df_original, df_capped):
    
    print("\nVerifying preprocessing...")

    # Labels must be identical
    assert (df_original["label"].values ==
            df_capped["label"].values).all(), \
        "ERROR: Labels changed during preprocessing!"
    print("  ✓ Labels unchanged")

    # Shape must be identical
    assert df_original.shape == df_capped.shape, \
        "ERROR: Shape changed during preprocessing!"
    print("  ✓ Shape unchanged")

    # Check std reduced for all features
    print("\n  Feature statistics comparison (all stations combined):")
    print(f"  {'Feature':<18} {'Orig min':>10} {'Cap min':>10} "
          f"{'Orig max':>10} {'Cap max':>10} "
          f"{'Orig std':>10} {'Cap std':>10}")
    print("  " + "-"*72)

    for col in FEATURE_COLS:
        o_min = df_original[col].min()
        o_max = df_original[col].max()
        o_std = df_original[col].std()
        c_min = df_capped[col].min()
        c_max = df_capped[col].max()
        c_std = df_capped[col].std()
        print(f"  {col:<18} {o_min:>10.3f} {c_min:>10.3f} "
              f"{o_max:>10.3f} {c_max:>10.3f} "
              f"{o_std:>10.3f} {c_std:>10.3f}")

    print("\n Verification complete")


# Visualise 

def visualise(df_original, df_capped):
    """
    Plot before and after distributions for all 6 features.
    Shows the effect of percentile capping on each feature.
    """
    print("\nGenerating visualisation...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    colors = {
        "before": "#5B8DB8",
        "after":  "#C0392B",
    }

    for idx, col in enumerate(FEATURE_COLS):
        ax = axes[idx]

        orig_vals   = df_original[col].dropna()
        capped_vals = df_capped[col].dropna()

        # Plot both distributions
        ax.hist(orig_vals, bins=60, alpha=0.5,
                color=colors["before"],
                label="Before capping",
                density=True)
        ax.hist(capped_vals, bins=60, alpha=0.5,
                color=colors["after"],
                label="After capping",
                density=True)

        # Add cap boundary lines
        lower_cap = np.percentile(df_original[col], LOWER_PCT)
        upper_cap = np.percentile(df_original[col], UPPER_PCT)
        ax.axvline(lower_cap, color="#2E7D52", linewidth=1.2,
                   linestyle="--",
                   label=f"{LOWER_PCT}th pct cap")
        ax.axvline(upper_cap, color="#C8922A", linewidth=1.2,
                   linestyle="--",
                   label=f"{UPPER_PCT}th pct cap")

        ax.set_title(col, fontsize=11, fontweight="bold",
                     color="#1B3A5C")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_facecolor("#F7FAFD")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    plt.suptitle(
        "Step 5 — Preprocessing: Percentile Capping "
        f"({LOWER_PCT}th – {UPPER_PCT}th percentile)\n"
        "Before vs After — all stations combined",
        fontsize=12, fontweight="bold",
        color="#1B3A5C", y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "step5_preprocessing.png")
    plt.savefig(out_path, dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved: {out_path}")


# Main 

def main():
    print("=" * 55)
    print("STEP 5 — PREPROCESSING AND OUTLIER HANDLING")
    print("=" * 55)
    print("NOTE: Scaling is intentionally deferred to Step 6")
    print("      (fitted on training data only — no leakage)\n")

    #Load data
    df = load_data()
    if df is None:
        return

    #Check missing values
    df = check_missing(df)

    #Apply percentile capping
    df_original = df.copy()
    df_capped, caps_df = percentile_capping(df)

    #Verify preprocessing
    verify_preprocessing(df_original, df_capped)

    #Visualise before/after
    visualise(df_original, df_capped)

    #Save outputs
    # Preprocessed dataset
    out_path = os.path.join(OUTPUT_DIR, "vlf_preprocessed.csv")
    df_capped.to_csv(out_path, index=False)
    print(f"\n  Preprocessed dataset saved : {out_path}")
    print(f"  Shape : {df_capped.shape[0]:,} rows × "
          f"{df_capped.shape[1]} columns")

    # Cap values record — important for thesis documentation
    caps_path = os.path.join(OUTPUT_DIR, "step5_outlier_caps.csv")
    caps_df.to_csv(caps_path, index=False)
    print(f"  Cap values saved           : {caps_path}")
    print("  (Keep this file — documents exactly what capping was applied)")

    print("\n Complete")
  


if __name__ == "__main__":
    main()