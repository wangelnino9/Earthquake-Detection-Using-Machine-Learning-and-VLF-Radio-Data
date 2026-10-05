import pandas as pd
import numpy as np
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.naive_bayes   import GaussianNB

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE  = 42
SEQ_LEN       = 42
N_FEATURES    = 4
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 0.001

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

FEATURE_COLS = ["Amp_Mean", "Amp_Std", "Phase_Mean", "Phase_Std"]


# Model Definitions

class SimpleNN(nn.Module):
    def __init__(self, input_size=N_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(1)


class DeepNN(nn.Module):
    def __init__(self, input_size=N_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(1)


class CNN1D(nn.Module):
    def __init__(self, n_features=N_FEATURES, seq_len=SEQ_LEN):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=6, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=6, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.classifier(self.conv_block(x)).squeeze(1)


class LSTMModel(nn.Module):
    def __init__(self, input_size=N_FEATURES,
                 hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(1)


#Load Data

def load_data():
    print("Loading data...")

    required = ["X_train.csv", "y_train.csv",
                "X_test.csv",  "y_test.csv",
                "meta_train.csv", "meta_test.csv",
                "X_val.csv", "y_val.csv", "meta_val.csv"]
    for f in required:
        if not os.path.exists(os.path.join(OUTPUT_DIR, f)):
            print(f"  ERROR: {f} not found.")
            return None

    eq_path = os.path.join(OUTPUT_DIR, "earthquake_days.csv")
    if not os.path.exists(eq_path):
        print("  ERROR: earthquake_days.csv not found.")
        return None

    X_train    = pd.read_csv(os.path.join(OUTPUT_DIR, "X_train.csv")).values
    y_train    = pd.read_csv(os.path.join(OUTPUT_DIR, "y_train.csv")).values.ravel()
    X_test     = pd.read_csv(os.path.join(OUTPUT_DIR, "X_test.csv")).values
    y_test     = pd.read_csv(os.path.join(OUTPUT_DIR, "y_test.csv")).values.ravel()
    X_val      = pd.read_csv(os.path.join(OUTPUT_DIR, "X_val.csv")).values
    y_val      = pd.read_csv(os.path.join(OUTPUT_DIR, "y_val.csv")).values.ravel()
    meta_train = pd.read_csv(os.path.join(OUTPUT_DIR, "meta_train.csv"))
    meta_test  = pd.read_csv(os.path.join(OUTPUT_DIR, "meta_test.csv"))
    meta_val   = pd.read_csv(os.path.join(OUTPUT_DIR, "meta_val.csv"))
    eq_days    = pd.read_csv(eq_path)
    X_train_raw = pd.read_csv(
    os.path.join(OUTPUT_DIR, "X_train_raw.csv")).values
    y_train_raw = pd.read_csv(
    os.path.join(OUTPUT_DIR, "y_train_raw.csv")).values.ravel()

    print(f"  X_train : {X_train.shape}  (scaled + SMOTE)")
    print(f"  y_train : {y_train.shape}  "
          f"(EQ={y_train.sum():,}, no-EQ={(y_train==0).sum():,})")
    print(f"  X_val   : {X_val.shape}   (real, no SMOTE)")
    print(f"  X_test  : {X_test.shape}   (scaled, real)")
    print(f"  EQ days : {len(eq_days):,} earthquake events")
    
    return (X_train, y_train, X_test, y_test,
        X_val, y_val,
        X_train_raw, y_train_raw,
        meta_train, meta_test, meta_val,
        eq_days)


# Sequence Builder 1: Sliding Window

def build_sequences_sliding(X, y, meta, seq_len=SEQ_LEN):
    print(f"\nBuilding sliding window sequences...")

    meta      = meta.copy().reset_index(drop=True)
    meta["date"] = pd.to_datetime(meta["date"])
    sort_cols = ["date", "window_id"] \
                if "window_id" in meta.columns else ["date"]

    X_sequences = []
    y_sequences = []

    for stn in sorted(meta["station"].unique()):
        stn_mask = meta["station"] == stn
        stn_idx  = meta[stn_mask].sort_values(
            sort_cols).index.tolist()
        X_stn = X[stn_idx]
        y_stn = y[stn_idx]
        n     = len(stn_idx)

        for i in range(n - seq_len + 1):
            X_sequences.append(X_stn[i : i + seq_len])
            y_sequences.append(y_stn[i + seq_len - 1])

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)

    print(f"  Sequences : {X_seq.shape}")
    print(f"  EQ={y_seq.sum():,}, "
          f"no-EQ={(y_seq==0).sum():,} "
          f"({y_seq.mean()*100:.1f}% EQ)")

    return X_seq, y_seq


# Sequence Builder 2: Event-Anchored

def build_sequences_event(X, y, meta, eq_days_df,
                           seq_len=SEQ_LEN, mode="train"):

    meta = meta.copy().reset_index(drop=True)
    meta["date"] = pd.to_datetime(meta["date"])
    sort_cols = ["date", "window_id"] \
                if "window_id" in meta.columns else ["date"]

    # Build EQ lookup
    eq_lookup = {}
    for _, row in eq_days_df.iterrows():
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

    pos_X = []
    neg_X = []

    for stn in sorted(meta["station"].unique()):
        stn_mask = meta["station"] == stn
        stn_meta = meta[stn_mask].sort_values(sort_cols)
        stn_idx  = stn_meta.index.tolist()
        X_stn    = X[stn_idx]
        y_stn    = y[stn_idx]
        dates_str = stn_meta["date"].dt.strftime(
            "%Y-%m-%d").values
        n = len(stn_idx)

        #Positive: ONE sequence per earthquake
        for eq_date_str, affected_stns in eq_lookup.items():
            if stn not in affected_stns:
                continue
            positions = np.where(dates_str == eq_date_str)[0]
            if len(positions) == 0:
                continue
            # Last window of earthquake day
            last_pos  = positions[-1]
            start_pos = last_pos - seq_len + 1
            if start_pos < 0:
                continue
            X_window = X_stn[start_pos : last_pos + 1]
            if len(X_window) == seq_len:
                pos_X.append(X_window)
        i = 0
        while i + seq_len <= n:
            window_y = y_stn[i : i + seq_len]
            if window_y.sum() == 0:  # all rows normal
                X_window = X_stn[i : i + seq_len]
                if len(X_window) == seq_len:
                    neg_X.append(X_window)
                i += seq_len  
            else:
                i += 1  

    # Subsample negatives to match positives
    np.random.seed(RANDOM_STATE)
    n_pos = len(pos_X)
    n_neg = len(neg_X)

    print(f"  Positive before balance : {n_pos:,}")
    print(f"  Negative before balance : {n_neg:,}")

    if n_pos == 0 or n_neg == 0:
        print(f"  WARNING: Empty sequences for {mode}")
        return np.array([]), np.array([])

    if n_neg > n_pos:
        idx   = np.random.choice(n_neg, size=n_pos, replace=False)
        neg_X = [neg_X[i] for i in idx]
    elif n_pos > n_neg:
        idx   = np.random.choice(n_pos, size=n_neg, replace=False)
        pos_X = [pos_X[i] for i in idx]

    X_sequences = pos_X + neg_X
    y_sequences = [1] * len(pos_X) + [0] * len(neg_X)

    shuffle_idx = np.random.permutation(len(X_sequences))
    X_sequences = [X_sequences[i] for i in shuffle_idx]
    y_sequences = [y_sequences[i] for i in shuffle_idx]

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)

    print(f"  Total sequences : {len(X_seq):,}")
    print(f"  Positive (EQ)   : {y_seq.sum():,}")
    print(f"  Negative (no EQ): {(y_seq==0).sum():,}")
    print(f"  Shape           : {X_seq.shape}")

    return X_seq, y_seq


#PyTorch Training

def train_pytorch_model(model, X_train_t, y_train_t,
                         model_name, epochs=EPOCHS,
                         X_val_t=None, y_val_t=None):

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model  = model.to(device)

    n_pos      = y_train_t.sum().item()
    n_neg      = len(y_train_t) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos],
                               dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=LEARNING_RATE)

    dataset    = TensorDataset(X_train_t.to(device),
                                y_train_t.to(device))
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

    X_vl = X_val_t.to(device) if X_val_t is not None else None
    y_vl = y_val_t.to(device) if y_val_t is not None else None
    model.train()
    print(f"    Training on {device}...")

    loss_history  = []
    val_history   = []
    best_val_loss = float('inf')
    patience      = 30
    no_improve    = 0
    best_epoch    = 0

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        model.eval()
        with torch.no_grad():
            if X_vl is not None:
                val_logits = model(X_vl)
                val_loss   = criterion(
                    val_logits, y_vl.float()).item()
            else:
                val_loss = 0.0
        val_history.append(val_loss)
        model.train()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:>3}/{epochs} — "
                  f"Train Loss: {avg_loss:.4f}  "
                  f"Val Loss: {val_loss:.4f}")

        # Early stopping
        if X_vl is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch + 1
                torch.save(model.state_dict(),
                    os.path.join(MODELS_DIR,
                                 f"{model_name}_best.pt"))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stopping at epoch "
                          f"{epoch+1} — "
                          f"best epoch: {best_epoch} "
                          f"(val={best_val_loss:.4f})")
                    break

    # Load best model before returning
    if X_vl is not None:
        best_path = os.path.join(MODELS_DIR,
                                  f"{model_name}_best.pt")
        if os.path.exists(best_path):
            model.load_state_dict(
                torch.load(best_path, map_location="cpu"))
            print(f"    Best model restored "
                  f"from epoch {best_epoch}")

    model.eval()
    return model, loss_history, val_history


#Train Classical Models

def train_classical_models(X_train, y_train):
    print("\n" + "="*55)
    print("CLASSICAL MODELS (7)")
    print("="*55)

    models = {
        "01_LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000),
        "02_DecisionTree":       DecisionTreeClassifier(
            random_state=RANDOM_STATE),
        "03_RandomForest":       RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1),
        "04_GradientBoosting":   GradientBoostingClassifier(
            random_state=RANDOM_STATE),
        "05_SVM":                SVC(
            probability=True, random_state=RANDOM_STATE),
        "06_KNN":                KNeighborsClassifier(n_jobs=-1),
        "07_NaiveBayes":         GaussianNB(),
    }

    trained = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        trained[name] = model
        print(f"  ✓ Done in {elapsed:.1f}s — saved to {path}")

    return trained


# Train Deep Learning Models

def train_deep_learning_models(X_train_flat, y_train_flat,
                                X_train_seq,  y_train_seq,
                                X_val_flat,   y_val_flat,
                                X_val_seq,    y_val_seq):

    print("\n" + "="*55)
    print("DEEP LEARNING MODELS (4) — PyTorch")
    print("="*55)
    print("  SimpleNN/DeepNN : flat SMOTE input")
    print("  CNN1D/LSTM      : event-anchored sequences")

    X_flat_t     = torch.tensor(X_train_flat, dtype=torch.float32)
    y_flat_t     = torch.tensor(y_train_flat, dtype=torch.float32)
    X_seq_t      = torch.tensor(X_train_seq,  dtype=torch.float32)
    y_seq_t      = torch.tensor(y_train_seq,  dtype=torch.float32)
    X_val_flat_t = torch.tensor(X_val_flat,   dtype=torch.float32)
    y_val_flat_t = torch.tensor(y_val_flat,   dtype=torch.float32)
    X_val_seq_t  = torch.tensor(X_val_seq,    dtype=torch.float32)
    y_val_seq_t  = torch.tensor(y_val_seq,    dtype=torch.float32)

    dl_models = {
        "08_SimpleNN": (SimpleNN(),  X_flat_t, y_flat_t,
                        X_val_flat_t, y_val_flat_t),
        "09_DeepNN":   (DeepNN(),    X_flat_t, y_flat_t,
                        X_val_flat_t, y_val_flat_t),
        "10_CNN1D":    (CNN1D(),     X_seq_t,  y_seq_t,
                        X_val_seq_t,  y_val_seq_t),
        "11_LSTM":     (LSTMModel(), X_seq_t,  y_seq_t,
                        X_val_seq_t,  y_val_seq_t),
    }

    trained       = {}
    all_histories = {}
    val_histories = {}

    for name, (model, X_t, y_t, X_v, y_v) in dl_models.items():
        print(f"\n  Training {name}...")
        print(f"    Train shape : {tuple(X_t.shape)}")
        print(f"    Val shape   : {tuple(X_v.shape)}")
        start = time.time()

        trained_model, loss_hist, val_hist = train_pytorch_model(
            model, X_t, y_t, name,
            epochs=EPOCHS,
            X_val_t=X_v,
            y_val_t=y_v
        )
        elapsed = time.time() - start

        path = os.path.join(MODELS_DIR, f"{name}.pt")
        torch.save(trained_model.state_dict(), path)
        arch_path = os.path.join(MODELS_DIR, f"{name}_arch.pkl")
        with open(arch_path, "wb") as f:
            pickle.dump(trained_model, f)

        trained[name]       = trained_model
        all_histories[name] = loss_hist
        val_histories[name] = val_hist
        print(f"  ✓ Done in {elapsed:.1f}s — saved to {path}")

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "08_SimpleNN": "#2E6DA4",
        "09_DeepNN":   "#2E7D52",
        "10_CNN1D":    "#C0392B",
        "11_LSTM":     "#C8922A",
    }
    for name, hist in all_histories.items():
        lbl = name.split("_", 1)[1]
        ax.plot(range(1, len(hist)+1), hist,
                label=f"{lbl} train",
                color=colors.get(name, "gray"),
                linewidth=1.5, linestyle="-")
    for name, hist in val_histories.items():
        lbl = name.split("_", 1)[1]
        ax.plot(range(1, len(hist)+1), hist,
                label=f"{lbl} val",
                color=colors.get(name, "gray"),
                linewidth=1.5, linestyle="--")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(
        "Training and Validation Loss Curves\n"
        "Solid = Training  |  Dashed = Validation",
        fontsize=12, fontweight="bold", color="#1B3A5C")
    ax.legend(fontsize=9, ncol=2)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
    plt.savefig(curve_path, dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Loss curves saved: {curve_path}")

    return trained


# Main

def main():
    print("=" * 55)
    print("TRAIN 11 ML MODELS")
    print("=" * 55)
    print(f"  Random state  : {RANDOM_STATE}")
    print(f"  Seq len       : {SEQ_LEN} steps (42 = 7 days)")
    print(f"  DL Epochs     : {EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  CNN/LSTM      : event-anchored sequences")
    print(f"  Flat models   : SMOTE balanced")

    data = load_data()
    if data is None:
        return

    (X_train, y_train, X_test, y_test,
     X_val, y_val,
     X_train_raw, y_train_raw,
     meta_train, meta_test, meta_val,
     eq_days) = data

    # Test sequences — sliding window for evaluation
    print("\n--- Test sequences (sliding window) ---")
    X_test_seq, y_test_seq = build_sequences_sliding(
        X_test, y_test, meta_test, seq_len=SEQ_LEN)
    np.save(os.path.join(OUTPUT_DIR,
                          "X_test_seq.npy"), X_test_seq)
    np.save(os.path.join(OUTPUT_DIR,
                          "y_test_seq.npy"), y_test_seq)
    print(f"  Test sequences saved: {X_test_seq.shape}")

# Training sequences — event-anchored for CNN1D/LSTM
    print("\n--- Training sequences (event-anchored) ---")
    X_train_seq, y_train_seq = build_sequences_event(
        X_train_raw, y_train_raw, meta_train, eq_days,
        seq_len=SEQ_LEN, mode="train")

    # Validation sequences — event-anchored
    print("\n--- Validation sequences (event-anchored) ---")
    X_val_seq, y_val_seq = build_sequences_event(
        X_val, y_val, meta_val, eq_days,
        seq_len=SEQ_LEN, mode="val")

    # Check sequences were built successfully
    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        print("\nERROR: Event-anchored sequences are empty.")
        print("Check earthquake_days.csv and meta_train dates.")
        return

    # Train classical models
    classical_models = train_classical_models(X_train, y_train)

    # Train deep learning models
    dl_models = train_deep_learning_models(
        X_train,     y_train,
        X_train_seq, y_train_seq,
        X_val,       y_val,
        X_val_seq,   y_val_seq
    )

    # Summary
    all_models = (list(classical_models.keys()) +
                  list(dl_models.keys()))
    print("\n" + "="*55)
    print("TRAINING COMPLETE — ALL 11 MODELS")
    print("="*55)
    print(f"\n  {'#':<4} {'Model':<30} {'Format':<12} {'Saved'}")
    print("  " + "-"*60)
    for i, name in enumerate(all_models, 1):
        fmt    = "Sequence" if name in ["10_CNN1D", "11_LSTM"] \
                 else "Flat"
        ext    = ".pt" if i >= 8 else ".pkl"
        path   = os.path.join(MODELS_DIR, f"{name}{ext}")
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {i:<4} {name:<30} {fmt:<12} {exists} {path}")

    log_path = os.path.join(OUTPUT_DIR, "training_log.txt")
    with open(log_path, "w") as f:
        f.write("Training Log\n")
        f.write("="*40 + "\n")
        f.write(f"X_train flat   : {X_train.shape}\n")
        f.write(f"X_train seq    : {X_train_seq.shape}\n")
        f.write(f"X_val seq      : {X_val_seq.shape}\n")
        f.write(f"X_test seq     : {X_test_seq.shape}\n")
        f.write(f"Seq length     : {SEQ_LEN}\n")
        f.write(f"DL epochs      : {EPOCHS}\n")
        f.write(f"CNN/LSTM method: event-anchored\n")
        f.write(f"Models trained : {len(all_models)}\n")
        for name in all_models:
            f.write(f"  {name}\n")
    print(f"\n  Log saved: {log_path}")



if __name__ == "__main__":
    main()