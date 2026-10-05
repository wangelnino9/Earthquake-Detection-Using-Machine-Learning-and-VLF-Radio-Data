import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
SEQ_LEN      = 42
N_FEATURES   = 4
TEST_SIZE    = 0.20
VAL_SIZE     = 0.10

FEATURE_COLS = ["Amp_Mean", "Amp_Std",
                "Phase_Mean", "Phase_Std"]

np.random.seed(RANDOM_STATE)


#Load full dataset

def load_data():
    print("Loading full dataset...")

    data_path = os.path.join(OUTPUT_DIR, "vlf_preprocessed.csv")
    eq_path   = os.path.join(OUTPUT_DIR, "earthquake_days.csv")

    if not os.path.exists(data_path):
        print(f"  ERROR: {data_path} not found.")
        return None, None

    if not os.path.exists(eq_path):
        print(f"  ERROR: {eq_path} not found.")
        return None, None

    df      = pd.read_csv(data_path)
    eq_days = pd.read_csv(eq_path)

    df["date"] = pd.to_datetime(df["date"])

    sort_cols = ["date", "window_id"] \
                if "window_id" in df.columns else ["date"]
    df = df.sort_values(["station"] + sort_cols)\
           .reset_index(drop=True)

    print(f"  Total rows     : {len(df):,}")
    print(f"  Unique dates   : {df['date'].nunique():,}")
    print(f"  Stations       : {sorted(df['station'].unique())}")
    print(f"  Label=1 (EQ)   : {df['label'].sum():,} "
          f"({df['label'].mean()*100:.1f}%)")
    print(f"  Label=0        : {(df['label']==0).sum():,}")
    print(f"  EQ events      : {len(eq_days):,}")

    return df, eq_days


# Build earthquake lookup

def build_eq_lookup(eq_days):
    eq_lookup = {}
    for _, row in eq_days.iterrows():
        eq_date = str(pd.to_datetime(row["date"]).date())
        paths   = str(row["matched_paths"])
        if not paths or paths == "nan":
            continue
        stations = [s.strip() for s in
                    paths.replace("|", ",").split(",")
                    if s.strip()]
        if eq_date not in eq_lookup:
            eq_lookup[eq_date] = []
        eq_lookup[eq_date].extend(stations)
    return eq_lookup


#Build all sequences

def build_all_sequences(df, eq_lookup, seq_len=SEQ_LEN):
    print("\n" + "="*55)
    print("BUILDING ALL SEQUENCES")
    print("="*55)
    print(f"  Seq length : {seq_len} steps "
          f"(7 days × 6 windows)")
    print(f"  Positive   : one per EQ per station")
    print(f"  Negative   : non-overlapping step={seq_len}")

    sort_cols = ["date", "window_id"] \
                if "window_id" in df.columns else ["date"]

    pos_X    = []
    neg_X    = []
    pos_meta = []
    neg_meta = []

    for stn in sorted(df["station"].unique()):
        stn_df    = df[df["station"] == stn]\
                    .sort_values(sort_cols)\
                    .reset_index(drop=True)
        X_stn     = stn_df[FEATURE_COLS].values
        y_stn     = stn_df["label"].values
        dates_str = stn_df["date"].dt.strftime(
                    "%Y-%m-%d").values
        n         = len(stn_df)

        #Positive sequences
        for eq_date_str, affected_stns in eq_lookup.items():
            if stn not in affected_stns:
                continue
            positions = np.where(
                dates_str == eq_date_str)[0]
            if len(positions) == 0:
                continue
            last_pos  = positions[-1]
            start_pos = last_pos - seq_len + 1
            if start_pos < 0:
                continue
            X_window = X_stn[start_pos : last_pos + 1]
            if len(X_window) == seq_len:
                pos_X.append(X_window)
                pos_meta.append({
                    "station"  : stn,
                    "eq_date"  : eq_date_str,
                    "end_date" : dates_str[last_pos],
                    "label"    : 1
                })

        #Negative sequences
        i = 0
        while i + seq_len <= n:
            window_y = y_stn[i : i + seq_len]
            if window_y.sum() == 0:
                X_window = X_stn[i : i + seq_len]
                if len(X_window) == seq_len:
                    neg_X.append(X_window)
                    neg_meta.append({
                        "station"  : stn,
                        "eq_date"  : "none",
                        "end_date" : dates_str[i + seq_len - 1],
                        "label"    : 0
                    })
                i += seq_len
            else:
                i += 1

    n_pos = len(pos_X)
    n_neg = len(neg_X)

    print(f"\n  Positive before balance : {n_pos:,}")
    print(f"  Negative before balance : {n_neg:,}")

    # Balance by sampling
    if n_neg > n_pos:
        idx   = np.random.choice(
            n_neg, size=n_pos, replace=False)
        neg_X    = [neg_X[i]    for i in idx]
        neg_meta = [neg_meta[i] for i in idx]
    elif n_pos > n_neg:
        idx   = np.random.choice(
            n_pos, size=n_neg, replace=False)
        pos_X    = [pos_X[i]    for i in idx]
        pos_meta = [pos_meta[i] for i in idx]

    X_all    = np.array(pos_X + neg_X,
                        dtype=np.float32)
    y_all    = np.array([1]*len(pos_X) +
                        [0]*len(neg_X),
                        dtype=np.int32)
    meta_all = pd.DataFrame(pos_meta + neg_meta)

    # Shuffle
    shuffle_idx = np.random.permutation(len(X_all))
    X_all       = X_all[shuffle_idx]
    y_all       = y_all[shuffle_idx]
    meta_all    = meta_all.iloc[shuffle_idx]\
                           .reset_index(drop=True)

    print(f"\n  Total sequences : {len(X_all):,}")
    print(f"  Positive (EQ)   : {y_all.sum():,}")
    print(f"  Negative        : {(y_all==0).sum():,}")
    print(f"  Shape           : {X_all.shape}")

    return X_all, y_all, meta_all


#Split sequences

def split_sequences(X_all, y_all, meta_all):
    print("\n" + "="*55)
    print("SPLITTING SEQUENCES")
    print("="*55)


    idx = np.arange(len(X_all))
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all
    )

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all[idx_trainval]
    )

    X_train = X_all[idx_train]
    y_train = y_all[idx_train]
    X_val   = X_all[idx_val]
    y_val   = y_all[idx_val]
    X_test  = X_all[idx_test]
    y_test  = y_all[idx_test]

    print(f"\n  Total sequences  : {len(X_all):,}")
    print(f"\n  Train (72%)      : {len(X_train):,} sequences")
    print(f"    EQ={y_train.sum():,}, "
          f"no-EQ={(y_train==0).sum():,}")
    print(f"\n  Val   (8%)       : {len(X_val):,} sequences")
    print(f"    EQ={y_val.sum():,}, "
          f"no-EQ={(y_val==0).sum():,}")
    print(f"\n  Test  (20%)      : {len(X_test):,} sequences")
    print(f"    EQ={y_test.sum():,}, "
          f"no-EQ={(y_test==0).sum():,}")

    return (X_train, y_train,
            X_val,   y_val,
            X_test,  y_test)


# Scale sequences 
def scale_sequences(X_train, X_val, X_test):
    print("\n" + "="*55)
    print("SCALING SEQUENCES")
    print("="*55)
    print("  Fitting StandardScaler on TRAIN only...")

    n_train, seq_len, n_feat = X_train.shape

    # Flatten to 2D for scaler fitting
    X_train_2d = X_train.reshape(-1, n_feat)
    X_val_2d   = X_val.reshape(-1, n_feat)
    X_test_2d  = X_test.reshape(-1, n_feat)

    scaler = StandardScaler()
    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(
        X_train_2d).reshape(X_train.shape)
    X_val_scaled   = scaler.transform(
        X_val_2d).reshape(X_val.shape)
    X_test_scaled  = scaler.transform(
        X_test_2d).reshape(X_test.shape)

    print(f"\n  Scaler stats (fitted on train rows):")
    print(f"  {'Feature':<18} {'Mean':>10} {'Std':>10}")
    print("  " + "-"*40)
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col:<18} "
              f"{scaler.mean_[i]:>10.4f} "
              f"{scaler.scale_[i]:>10.4f}")

    scaler_path = os.path.join(
        OUTPUT_DIR, "seq_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Scaler saved: {scaler_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled


#Flatten for classical models

def flatten_sequences(X_train, X_val, X_test):
    print("\n" + "="*55)
    print("FLATTENING FOR CLASSICAL MODELS")
    print("="*55)

    X_train_flat = X_train.reshape(
        X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(
        X_val.shape[0], -1)
    X_test_flat  = X_test.reshape(
        X_test.shape[0], -1)

    print(f"  Sequence shape  : {X_train.shape}")
    print(f"  Flattened shape : {X_train_flat.shape}")
    print(f"  Features per seq: {X_train_flat.shape[1]}")
    print(f"  = {SEQ_LEN} steps × {N_FEATURES} features")

    # Generate feature names for importance analysis
    feature_names = []
    for step in range(SEQ_LEN):
        day = 6 - (step // 6)
        win = step % 6
        for feat in FEATURE_COLS:
            feature_names.append(
                f"Day-{day}_W{win}_{feat}")

    fname_path = os.path.join(
        OUTPUT_DIR, "feature_names.txt")
    with open(fname_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")
    print(f"  Feature names saved: {fname_path}")

    return X_train_flat, X_val_flat, X_test_flat


#Save outputs

def save_outputs(X_train, y_train,
                 X_val,   y_val,
                 X_test,  y_test,
                 X_train_flat, X_val_flat, X_test_flat):
    print("\nSaving outputs...")

    files = {
        "X_train_seq.npy"  : X_train,
        "y_train_seq.npy"  : y_train,
        "X_val_seq.npy"    : X_val,
        "y_val_seq.npy"    : y_val,
        "X_test_seq.npy"   : X_test,
        "y_test_seq.npy"   : y_test,
        "X_train_flat.npy" : X_train_flat,
        "X_val_flat.npy"   : X_val_flat,
        "X_test_flat.npy"  : X_test_flat,
    }

    for fname, data in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        np.save(path, data)
        print(f"  {fname:<22} : {data.shape}")

    # Save log
    log_path = os.path.join(
        OUTPUT_DIR, "sequence_log.txt")
    with open(log_path, "w") as f:
        f.write("Sequence Build Log\n")
        f.write("="*40 + "\n")
        f.write(f"SEQ_LEN      : {SEQ_LEN}\n")
        f.write(f"N_FEATURES   : {N_FEATURES}\n")
        f.write(f"TEST_SIZE    : {TEST_SIZE}\n")
        f.write(f"VAL_SIZE     : {VAL_SIZE}\n")
        f.write(f"RANDOM_STATE : {RANDOM_STATE}\n")
        f.write(f"\nTrain seq    : {X_train.shape}\n")
        f.write(f"Val seq      : {X_val.shape}\n")
        f.write(f"Test seq     : {X_test.shape}\n")
        f.write(f"Train flat   : {X_train_flat.shape}\n")
        f.write(f"\nSplit method : sequence level\n")
        f.write(f"Sequences    : event-anchored, "
                f"no sliding window\n")
    print(f"  Log saved: {log_path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("BUILD SEQUENCES — NEW PIPELINE")
    print("=" * 55)
    print(f"  Seq length   : {SEQ_LEN} "
          f"(7 days × 6 windows)")
    print(f"  Split        : sequence level")
    print(f"  Train/Val/Test: "
          f"{int((1-TEST_SIZE)*(1-VAL_SIZE)*100)}% / "
          f"{int((1-TEST_SIZE)*VAL_SIZE*100)}% / "
          f"{int(TEST_SIZE*100)}%")
    print(f"  Sliding window: NO — independent only")

    # Load
    df, eq_days = load_data()
    if df is None:
        return

    # Build EQ lookup
    eq_lookup = build_eq_lookup(eq_days)

    # Build all sequences
    X_all, y_all, meta_all = build_all_sequences(
        df, eq_lookup, seq_len=SEQ_LEN)

    # Split at sequence level
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test) = split_sequences(
        X_all, y_all, meta_all)

    # Scale (fit on train only)
    (X_train, X_val,
     X_test) = scale_sequences(
        X_train, X_val, X_test)

    # Flatten for classical models
    (X_train_flat, X_val_flat,
     X_test_flat) = flatten_sequences(
        X_train, X_val, X_test)

    # Save all outputs
    save_outputs(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_train_flat,
        X_val_flat,
        X_test_flat)

    print("\n" + "="*55)
    print("Build sequences complete")
    print("="*55)
    print(f"\n  Train : {X_train.shape} sequences")
    print(f"  Val   : {X_val.shape} sequences")
    print(f"  Test  : {X_test.shape} sequences")
    print(f"  Flat  : {X_train_flat.shape} "
          f"(for classical models)")


if __name__ == "__main__":
    main()