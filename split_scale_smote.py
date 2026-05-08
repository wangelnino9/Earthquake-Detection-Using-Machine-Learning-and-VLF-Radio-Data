"""
Step 6 — Train/Test Split + Standard Scaling + SMOTE
======================================================
Three operations in strict order to prevent data leakage:

    1. Stratified 80/20 split at DATE LEVEL
       - Unique dates are split, not individual rows
       - All 8 stations for a given date stay together
       - A date is classified as earthquake day if ANY station has label=1
       - Prevents same earthquake event appearing in both train and test

    2. Standard Scaling — fitted on TRAINING data only
       - Scaler learns mean and std from training set
       - Same scaler applied to test set
       - Test set never influences the scaler

    3. SMOTE — applied to TRAINING data only
       - Generates synthetic minority class examples
       - Test set left completely untouched
       - Preserves real class distribution in test set

Input:
    outputs/vlf_preprocessed.csv     — from Step 5

Output:
    outputs/X_train.csv              — training features (scaled + SMOTE)
    outputs/X_test.csv               — test features (scaled, no SMOTE)
    outputs/y_train.csv              — training labels (after SMOTE)
    outputs/y_test.csv               — test labels (original distribution)
    outputs/step6_scaler.pkl         — fitted scaler (for future use)
    outputs/step6_split_summary.png  — visualisation of split and balance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "Amp_Mean",
    "Amp_Std",
    "Night_Amp_Mean",
    "Night_Amp_Std",
    "Phase_Mean",
    "Phase_Std",
]

RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80/20 split


# ── Step 1: Load Preprocessed Data ───────────────────────────

def load_data():
    """Load preprocessed dataset from Step 5."""
    print("Loading preprocessed dataset...")

    path = os.path.join(OUTPUT_DIR, "vlf_preprocessed.csv")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        print("  Please run Step 5 first.")
        return None

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    print(f"  Rows loaded      : {len(df):,}")
    print(f"  Unique dates     : {df['date'].nunique():,}")
    print(f"  Stations         : {sorted(df['station'].unique())}")
    print(f"  Label = 1 (EQ)   : {df['label'].sum():,} "
          f"({df['label'].mean()*100:.1f}%)")
    print(f"  Label = 0 (no EQ): {(df['label']==0).sum():,} "
          f"({(1-df['label'].mean())*100:.1f}%)")
    return df


# ── Step 2: Date Level Stratified Split ───────────────────────

def date_level_split(df):
    """
    Split at date level — not row level — to prevent data leakage.

    Logic:
        1. Get all unique dates
        2. For each date — classify as earthquake date if ANY station
           on that date has label = 1 (Option A)
        3. Stratified split of dates (80/20) preserving EQ date ratio
        4. Assign all rows for training dates → training set
           Assign all rows for test dates → test set

    This guarantees:
        - No earthquake event appears in both train and test
        - Class ratio is roughly preserved in both sets
        - All 8 stations for a given date stay together
    """
    print("\n" + "="*55)
    print("STEP 1: DATE LEVEL STRATIFIED SPLIT")
    print("="*55)

    # Get unique dates with their earthquake classification
    # A date = earthquake date if ANY station on that date has label=1
    date_labels = (df.groupby("date")["label"]
                   .max()   # max of 0s and 1s → 1 if any station is 1
                   .reset_index())
    date_labels.columns = ["date", "date_is_eq"]

    total_dates   = len(date_labels)
    eq_dates      = date_labels["date_is_eq"].sum()
    non_eq_dates  = total_dates - eq_dates

    print(f"\n  Total unique dates     : {total_dates:,}")
    print(f"  Earthquake dates       : {eq_dates:,} "
          f"({eq_dates/total_dates*100:.1f}%)")
    print(f"  Non-earthquake dates   : {non_eq_dates:,} "
          f"({non_eq_dates/total_dates*100:.1f}%)")

    # Stratified split of dates
    train_dates, test_dates = train_test_split(
        date_labels["date"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=date_labels["date_is_eq"]
    )

    train_dates = set(train_dates)
    test_dates  = set(test_dates)

    # Assign rows based on date membership
    df_train = df[df["date"].isin(train_dates)].copy()
    df_test  = df[df["date"].isin(test_dates)].copy()

    print(f"\n  Training set:")
    print(f"    Dates  : {len(train_dates):,} "
          f"({len(train_dates)/total_dates*100:.1f}%)")
    print(f"    Rows   : {len(df_train):,}")
    print(f"    EQ rows: {df_train['label'].sum():,} "
          f"({df_train['label'].mean()*100:.1f}%)")

    print(f"\n  Test set:")
    print(f"    Dates  : {len(test_dates):,} "
          f"({len(test_dates)/total_dates*100:.1f}%)")
    print(f"    Rows   : {len(df_test):,}")
    print(f"    EQ rows: {df_test['label'].sum():,} "
          f"({df_test['label'].mean()*100:.1f}%)")

    # Verify no date overlap
    overlap = train_dates & test_dates
    assert len(overlap) == 0, \
        f"ERROR: {len(overlap)} dates appear in both train and test!"
    print(f"\n  ✓ No date overlap between train and test")
    print(f"  ✓ Date level split complete")

    return df_train, df_test


# ── Step 3: Extract Features and Labels ──────────────────────

def extract_features_labels(df_train, df_test):
    """
    Separate features (X) and labels (y) for train and test.
    Also keeps metadata (date, station) separate — not used in training.
    """
    print("\n" + "="*55)
    print("STEP 2: EXTRACT FEATURES AND LABELS")
    print("="*55)

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["label"].values
    X_test  = df_test[FEATURE_COLS].values
    y_test  = df_test["label"].values

    # Keep metadata for reference
    meta_train = df_train[["date", "station"]].reset_index(drop=True)
    meta_test  = df_test[["date", "station"]].reset_index(drop=True)

    print(f"\n  X_train shape : {X_train.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  X_test shape  : {X_test.shape}")
    print(f"  y_test shape  : {y_test.shape}")
    print(f"  Features      : {FEATURE_COLS}")

    return X_train, y_train, X_test, y_test, meta_train, meta_test


# ── Step 4: Standard Scaling ──────────────────────────────────

def apply_scaling(X_train, X_test):
    """
    Fit StandardScaler on training data only.
    Apply the SAME fitted scaler to test data.

    StandardScaler formula for each feature:
        scaled_value = (value - mean) / std

    After scaling:
        training features: mean = 0, std = 1 (approximately)
        test features: scaled using training mean and std
                       (test mean will NOT be exactly 0 — that is correct)

    Critical: scaler.fit() is called ONLY on X_train.
              scaler.transform() is called on both X_train and X_test.
              NEVER call scaler.fit() on X_test.
    """
    print("\n" + "="*55)
    print("STEP 3: STANDARD SCALING")
    print("="*55)
    print("  Fitting scaler on TRAINING data only...")

    scaler = StandardScaler()

    # Fit on training data → learns mean and std from training only
    scaler.fit(X_train)

    # Transform both sets using training statistics
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\n  Scaler statistics (learned from training data):")
    print(f"  {'Feature':<18} {'Train mean':>12} {'Train std':>12}")
    print("  " + "-"*44)
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col:<18} {scaler.mean_[i]:>12.4f} "
              f"{scaler.scale_[i]:>12.4f}")

    # Verify scaling
    train_means = X_train_scaled.mean(axis=0)
    train_stds  = X_train_scaled.std(axis=0)
    print(f"\n  Verification — training set after scaling:")
    print(f"  {'Feature':<18} {'Mean (≈0)':>12} {'Std (≈1)':>12}")
    print("  " + "-"*44)
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col:<18} {train_means[i]:>12.6f} "
              f"{train_stds[i]:>12.6f}")

    print(f"\n  ✓ Scaling complete")
    print(f"  NOTE: Test set mean will NOT be exactly 0 — this is correct.")
    print(f"        Test data is scaled using training statistics only.")

    return X_train_scaled, X_test_scaled, scaler


# ── Step 5: SMOTE ─────────────────────────────────────────────

def apply_smote(X_train_scaled, y_train):
    """
    Apply SMOTE to training data only.
    Test data is NEVER touched by SMOTE.

    SMOTE (Synthetic Minority Over-Sampling Technique):
        For each minority class example (earthquake day):
            Find k nearest neighbours in feature space
            Generate synthetic examples along the line
            between the example and its neighbours

    sampling_strategy='auto' means:
        Oversample minority class until it equals majority class
        This gives approximately 50/50 balance in training

    random_state=42 ensures reproducibility.

    IMPORTANT:
        SMOTE is applied AFTER scaling.
        This is correct — SMOTE interpolates in the scaled feature space
        where distances between points are meaningful and comparable.
    """
    print("\n" + "="*55)
    print("STEP 4: SMOTE (TRAINING DATA ONLY)")
    print("="*55)

    n_before    = len(y_train)
    n_eq_before = y_train.sum()
    n_no_before = n_before - n_eq_before

    print(f"\n  Before SMOTE:")
    print(f"    Total training rows : {n_before:,}")
    print(f"    Label = 1 (EQ)      : {n_eq_before:,} "
          f"({n_eq_before/n_before*100:.1f}%)")
    print(f"    Label = 0 (no EQ)   : {n_no_before:,} "
          f"({n_no_before/n_before*100:.1f}%)")
    print(f"    Imbalance ratio     : 1 : "
          f"{n_no_before/n_eq_before:.1f}")

    print(f"\n  Applying SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_scaled, y_train
    )

    n_after    = len(y_train_smote)
    n_eq_after = y_train_smote.sum()
    n_no_after = n_after - n_eq_after
    n_synthetic = n_eq_after - n_eq_before

    print(f"\n  After SMOTE:")
    print(f"    Total training rows : {n_after:,}")
    print(f"    Label = 1 (EQ)      : {n_eq_after:,} "
          f"({n_eq_after/n_after*100:.1f}%)")
    print(f"    Label = 0 (no EQ)   : {n_no_after:,} "
          f"({n_no_after/n_after*100:.1f}%)")
    print(f"    Synthetic EQ rows   : {n_synthetic:,} generated")
    print(f"    Imbalance ratio     : 1 : "
          f"{n_no_after/n_eq_after:.1f}")

    print(f"\n  ✓ SMOTE complete")
    print(f"  NOTE: Test set untouched — "
          f"real distribution preserved for evaluation")

    return X_train_smote, y_train_smote


# ── Step 6: Visualise ─────────────────────────────────────────

def visualise(y_train_orig, y_train_smote, y_test, X_train_smote):
    """
    Four plots:
    1. Class distribution before/after SMOTE in training
    2. Class distribution in test set (unchanged)
    3. Feature distributions after scaling (training)
    4. SMOTE effect — original vs synthetic points (first 2 features)
    """
    print("\nGenerating visualisation...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    col_eq    = "#C0392B"
    col_no_eq = "#2E6DA4"
    col_syn   = "#C8922A"

    # ── Plot 1: Training set before/after SMOTE ───────────────
    ax = axes[0, 0]
    before_counts = [
        (y_train_orig == 0).sum(),
        (y_train_orig == 1).sum()
    ]
    after_counts = [
        (y_train_smote == 0).sum(),
        (y_train_smote == 1).sum()
    ]
    x = np.arange(2)
    w = 0.35
    bars1 = ax.bar(x - w/2, before_counts, width=w,
                   color=[col_no_eq, col_eq],
                   alpha=0.6, label="Before SMOTE",
                   edgecolor="white")
    bars2 = ax.bar(x + w/2, after_counts, width=w,
                   color=[col_no_eq, col_eq],
                   alpha=1.0, label="After SMOTE",
                   edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Label = 0\n(No EQ)", "Label = 1\n(EQ day)"])
    ax.set_title("Training Set: Before vs After SMOTE",
                 fontsize=11, fontweight="bold", color="#1B3A5C")
    ax.set_ylabel("Number of rows", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count in zip(list(bars1) + list(bars2),
                          before_counts + after_counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 50,
                f"{count:,}", ha="center", va="bottom",
                fontsize=8, color="#2D3748")

    # ── Plot 2: Test set distribution ─────────────────────────
    ax = axes[0, 1]
    test_counts = [(y_test == 0).sum(), (y_test == 1).sum()]
    bars = ax.bar(["Label = 0\n(No EQ)", "Label = 1\n(EQ day)"],
                  test_counts,
                  color=[col_no_eq, col_eq],
                  width=0.5, edgecolor="white")
    for bar, count in zip(bars, test_counts):
        pct = count / len(y_test) * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 10,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=9, color="#2D3748")
    ax.set_title("Test Set Distribution (Unchanged — Real World)",
                 fontsize=11, fontweight="bold", color="#1B3A5C")
    ax.set_ylabel("Number of rows", fontsize=10)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Plot 3: Feature distributions after scaling ───────────
    ax = axes[1, 0]
    for i, col in enumerate(FEATURE_COLS):
        ax.hist(X_train_smote[:, i], bins=50, alpha=0.5,
                label=col, density=True)
    ax.axvline(0, color="black", linewidth=1.0,
               linestyle="--", alpha=0.5, label="Mean=0")
    ax.set_title("Feature Distributions After Scaling\n(Training + SMOTE)",
                 fontsize=11, fontweight="bold", color="#1B3A5C")
    ax.set_xlabel("Scaled value", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Plot 4: Class balance summary ─────────────────────────
    ax = axes[1, 1]
    categories = ["Original\nTraining", "After SMOTE\nTraining",
                  "Test Set\n(Real)"]
    eq_pcts = [
        (y_train_orig == 1).mean() * 100,
        (y_train_smote == 1).mean() * 100,
        (y_test == 1).mean() * 100,
    ]
    colors_bar = [col_no_eq, col_eq, col_syn]
    bars = ax.bar(categories, eq_pcts,
                  color=colors_bar, width=0.5,
                  edgecolor="white")
    ax.axhline(50, color="gray", linewidth=1.0,
               linestyle="--", alpha=0.7, label="50% balance")
    for bar, pct in zip(bars, eq_pcts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#2D3748")
    ax.set_title("Earthquake Day % Across Sets",
                 fontsize=11, fontweight="bold", color="#1B3A5C")
    ax.set_ylabel("% earthquake day rows", fontsize=10)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=9)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Step 6 — Train/Test Split + Standard Scaling + SMOTE\n"
        "Date-level stratified split | "
        "Scaler fitted on training only | "
        "SMOTE on training only",
        fontsize=11, fontweight="bold",
        color="#1B3A5C", y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "step6_split_summary.png")
    plt.savefig(out_path, dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("STEP 6 — SPLIT + SCALING + SMOTE")
    print("=" * 55)
    print(f"  Random state : {RANDOM_STATE}")
    print(f"  Test size    : {TEST_SIZE*100:.0f}%")
    print(f"  Split level  : Date level (prevents leakage)")
    print(f"  Scaling      : StandardScaler (fit on train only)")
    print(f"  SMOTE        : Training data only")

    # 1. Load preprocessed data
    df = load_data()
    if df is None:
        return

    # 2. Date level stratified split
    df_train, df_test = date_level_split(df)

    # 3. Extract features and labels
    (X_train, y_train,
     X_test, y_test,
     meta_train, meta_test) = extract_features_labels(
        df_train, df_test
    )

    # Keep original y_train for visualisation
    y_train_orig = y_train.copy()

    # 4. Standard scaling — fit on train, apply to both
    X_train_scaled, X_test_scaled, scaler = apply_scaling(
        X_train, X_test
    )

    # 5. SMOTE — training only
    X_train_final, y_train_final = apply_smote(
        X_train_scaled, y_train
    )

    # 6. Visualise
    visualise(y_train_orig, y_train_final, y_test, X_train_final)

    # 7. Save all outputs
    print("\nSaving outputs...")

    # Training set (scaled + SMOTE)
    pd.DataFrame(X_train_final,
                 columns=FEATURE_COLS).to_csv(
        os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    pd.DataFrame(y_train_final,
                 columns=["label"]).to_csv(
        os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)

    # Test set (scaled, no SMOTE)
    pd.DataFrame(X_test_scaled,
                 columns=FEATURE_COLS).to_csv(
        os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    pd.DataFrame(y_test,
                 columns=["label"]).to_csv(
        os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

    # Metadata (date, station) for reference
    meta_train.to_csv(
        os.path.join(OUTPUT_DIR, "meta_train.csv"), index=False)
    meta_test.to_csv(
        os.path.join(OUTPUT_DIR, "meta_test.csv"), index=False)

    # Save fitted scaler for future use
    scaler_path = os.path.join(OUTPUT_DIR, "step6_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  X_train.csv  : {X_train_final.shape} "
          f"(scaled + SMOTE)")
    print(f"  y_train.csv  : {y_train_final.shape}")
    print(f"  X_test.csv   : {X_test_scaled.shape} "
          f"(scaled, real distribution)")
    print(f"  y_test.csv   : {y_test.shape}")
    print(f"  Scaler saved : {scaler_path}")

    print("\n✅ Step 6 complete.")
    print("   Next → Step 7: Train 11 ML models")


if __name__ == "__main__":
    main()