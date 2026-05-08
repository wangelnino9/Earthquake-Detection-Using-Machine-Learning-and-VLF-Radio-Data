"""
Step 7 — Train 11 ML Models
============================
Trains all 11 models on the preprocessed, scaled, SMOTE-balanced
training data from Step 6.

Classical models (Format 1 — flat 2D input):
    1.  Logistic Regression
    2.  Decision Tree
    3.  Random Forest
    4.  Gradient Boosting
    5.  Support Vector Machine (SVM)
    6.  K-Nearest Neighbours (KNN)
    7.  Naïve Bayes

Deep Learning models:
    8.  Simple Neural Network  (Format 1 — flat)
    9.  Deep Neural Network    (Format 1 — flat)
    10. 1D-CNN                 (Format 2 — sequence, shape: samples×7×6)
    11. LSTM                   (Format 2 — sequence, shape: samples×7×6)

All models use default hyperparameters (baseline run).
All models are saved to disk after training.

Input:
    outputs/X_train.csv    — training features (scaled + SMOTE)
    outputs/y_train.csv    — training labels
    outputs/X_test.csv     — test features (scaled)
    outputs/y_test.csv     — test labels
    outputs/meta_train.csv — date and station for training rows
    outputs/meta_test.csv  — date and station for test rows

Output:
    outputs/models/        — all 11 saved models
    outputs/step7_training_log.txt — training summary
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

# Sklearn classical models
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB

# PyTorch deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR  = "outputs"
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE   = 42
SEQ_LEN        = 7       # 7-day sliding window for LSTM and CNN
N_FEATURES     = 6       # Amp_Mean, Amp_Std, Night_Amp_Mean, Night_Amp_Std, Phase_Mean, Phase_Std
BATCH_SIZE     = 32
EPOCHS         = 50
LEARNING_RATE  = 0.001

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

FEATURE_COLS = [
    "Amp_Mean", "Amp_Std", "Night_Amp_Mean",
    "Night_Amp_Std", "Phase_Mean", "Phase_Std"
]


# ── Step 1: Load Data ─────────────────────────────────────────

def load_data():
    """Load training and test data from Step 6."""
    print("Loading data from Step 6...")

    required = ["X_train.csv", "y_train.csv",
                "X_test.csv",  "y_test.csv",
                "meta_train.csv", "meta_test.csv"]
    for f in required:
        if not os.path.exists(os.path.join(OUTPUT_DIR, f)):
            print(f"  ERROR: {f} not found. Run Step 6 first.")
            return None

    X_train    = pd.read_csv(os.path.join(OUTPUT_DIR, "X_train.csv")).values
    y_train    = pd.read_csv(os.path.join(OUTPUT_DIR, "y_train.csv")).values.ravel()
    X_test     = pd.read_csv(os.path.join(OUTPUT_DIR, "X_test.csv")).values
    y_test     = pd.read_csv(os.path.join(OUTPUT_DIR, "y_test.csv")).values.ravel()
    meta_train = pd.read_csv(os.path.join(OUTPUT_DIR, "meta_train.csv"))
    meta_test  = pd.read_csv(os.path.join(OUTPUT_DIR, "meta_test.csv"))

    print(f"  X_train : {X_train.shape}  (scaled + SMOTE)")
    print(f"  y_train : {y_train.shape}  "
          f"(EQ={y_train.sum():,}, no-EQ={(y_train==0).sum():,})")
    print(f"  X_test  : {X_test.shape}   (scaled, real distribution)")
    print(f"  y_test  : {y_test.shape}   "
          f"(EQ={y_test.sum():,}, no-EQ={(y_test==0).sum():,})")

    return X_train, y_train, X_test, y_test, meta_train, meta_test


# ── Step 2: Build Sequences for LSTM and CNN ──────────────────

def build_sequences(X, y, meta, seq_len=SEQ_LEN):
    """
    Build sliding window sequences per station for LSTM and CNN.

    For each station:
        Sort rows by date
        Create sequences of seq_len consecutive days
        Label = label of the LAST day in the sequence

    Example with seq_len=7:
        Sequence 1 = Days 1-7  → label of Day 7
        Sequence 2 = Days 2-8  → label of Day 8
        ...

    Only sequences within the same station are created.
    Never mix days from different stations.

    Returns:
        X_seq : shape (n_sequences, seq_len, n_features)
        y_seq : shape (n_sequences,)
    """
    print(f"\nBuilding {seq_len}-day sequences for LSTM/CNN...")

    meta = meta.copy().reset_index(drop=True)
    meta["date"] = pd.to_datetime(meta["date"])

    X_sequences = []
    y_sequences = []

    stations = sorted(meta["station"].unique())

    for stn in stations:
        # Get indices for this station sorted by date
        stn_mask = meta["station"] == stn
        stn_idx  = meta[stn_mask].sort_values("date").index.tolist()

        X_stn = X[stn_idx]
        y_stn = y[stn_idx]

        n = len(stn_idx)
        n_seq = n - seq_len + 1

        for i in range(n_seq):
            # Window of seq_len days
            X_window = X_stn[i : i + seq_len]   # shape: (seq_len, n_features)
            y_label  = y_stn[i + seq_len - 1]    # label of last day in window

            X_sequences.append(X_window)
            y_sequences.append(y_label)

    X_seq = np.array(X_sequences)  # shape: (n_seq, seq_len, n_features)
    y_seq = np.array(y_sequences)  # shape: (n_seq,)

    print(f"  Sequences built : {X_seq.shape}")
    print(f"  Labels          : EQ={y_seq.sum():,}, "
          f"no-EQ={(y_seq==0).sum():,} "
          f"({y_seq.mean()*100:.1f}% EQ)")

    return X_seq, y_seq


# ── PyTorch Model Definitions ─────────────────────────────────

class SimpleNN(nn.Module):
    """
    Simple Neural Network — 1 hidden layer.
    Input: flat features (batch, 6)
    """
    def __init__(self, input_size=N_FEATURES):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


class DeepNN(nn.Module):
    """
    Deep Neural Network — 3 hidden layers.
    Input: flat features (batch, 6)
    """
    def __init__(self, input_size=N_FEATURES):
        super(DeepNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network.
    Input: sequences (batch, seq_len, n_features)
    PyTorch Conv1d expects (batch, channels, length)
    so we permute to (batch, n_features, seq_len)
    """
    def __init__(self, n_features=N_FEATURES, seq_len=SEQ_LEN):
        super(CNN1D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=n_features,
                      out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)   # global average pooling
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)   # → (batch, n_features, seq_len)
        x = self.conv_block(x)
        x = self.classifier(x)
        return x.squeeze(1)


class LSTMModel(nn.Module):
    """
    LSTM Model.
    Input: sequences (batch, seq_len, n_features)
    Uses 2 LSTM layers with dropout for regularisation.
    """
    def __init__(self, input_size=N_FEATURES,
                 hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take output of last timestep only
        last_out = lstm_out[:, -1, :]   # (batch, hidden_size)
        return self.classifier(last_out).squeeze(1)


# ── PyTorch Training Helper ───────────────────────────────────

def train_pytorch_model(model, X_train_t, y_train_t,
                         model_name, epochs=EPOCHS):
    """
    Train a PyTorch model using binary cross-entropy loss
    and Adam optimiser.

    Uses class weights to handle remaining imbalance in sequences.
    Prints loss every 10 epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model  = model.to(device)

    # Class weights — inverse frequency
    n_pos  = y_train_t.sum().item()
    n_neg  = len(y_train_t) - n_pos
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

    model.train()
    print(f"    Training on {device}...")

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"    Epoch {epoch+1:>3}/{epochs} — "
                  f"Loss: {avg_loss:.4f}")

    model.eval()
    return model


# ── Train Classical Models ────────────────────────────────────

def train_classical_models(X_train, y_train):
    """
    Train 7 classical sklearn models.
    All use default hyperparameters for baseline run.
    """
    print("\n" + "="*55)
    print("CLASSICAL MODELS (7)")
    print("="*55)

    models = {
        "01_LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ),
        "02_DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "03_RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "04_GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "05_SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
        "06_KNN": KNeighborsClassifier(
            n_jobs=-1
        ),
        "07_NaiveBayes": GaussianNB(),
    }

    trained = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        # Save model
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)

        trained[name] = model
        print(f"  ✓ Done in {elapsed:.1f}s — saved to {path}")

    return trained


# ── Train Deep Learning Models ────────────────────────────────

def train_deep_learning_models(X_train_flat, y_train_flat,
                                X_train_seq, y_train_seq):
    """
    Train 4 PyTorch deep learning models.

    Flat input (SimpleNN, DeepNN):
        X shape: (n_samples, 6)

    Sequence input (CNN1D, LSTM):
        X shape: (n_samples, seq_len, 6)
    """
    print("\n" + "="*55)
    print("DEEP LEARNING MODELS (4) — PyTorch")
    print("="*55)

    # Convert to PyTorch tensors
    X_flat_t = torch.tensor(X_train_flat, dtype=torch.float32)
    y_flat_t = torch.tensor(y_train_flat, dtype=torch.float32)
    X_seq_t  = torch.tensor(X_train_seq,  dtype=torch.float32)
    y_seq_t  = torch.tensor(y_train_seq,  dtype=torch.float32)

    dl_models = {
        "08_SimpleNN": (SimpleNN(),    X_flat_t, y_flat_t),
        "09_DeepNN":   (DeepNN(),      X_flat_t, y_flat_t),
        "10_CNN1D":    (CNN1D(),       X_seq_t,  y_seq_t),
        "11_LSTM":     (LSTMModel(),   X_seq_t,  y_seq_t),
    }

    trained = {}
    for name, (model, X_t, y_t) in dl_models.items():
        print(f"\n  Training {name}...")
        print(f"    Input shape : {tuple(X_t.shape)}")
        start = time.time()

        trained_model = train_pytorch_model(
            model, X_t, y_t, name, epochs=EPOCHS
        )
        elapsed = time.time() - start

        # Save model
        path = os.path.join(MODELS_DIR, f"{name}.pt")
        torch.save(trained_model.state_dict(), path)

        # Also save model architecture for loading later
        arch_path = os.path.join(MODELS_DIR, f"{name}_arch.pkl")
        with open(arch_path, "wb") as f:
            pickle.dump(trained_model, f)

        trained[name] = trained_model
        print(f"  ✓ Done in {elapsed:.1f}s — saved to {path}")

    return trained


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("STEP 7 — TRAIN 11 ML MODELS")
    print("=" * 55)
    print(f"  Random state  : {RANDOM_STATE}")
    print(f"  Sequence len  : {SEQ_LEN} days (LSTM + CNN)")
    print(f"  DL Epochs     : {EPOCHS}")
    print(f"  DL Batch size : {BATCH_SIZE}")
    print(f"  DL LR         : {LEARNING_RATE}")

    # 1. Load data
    data = load_data()
    if data is None:
        return
    X_train, y_train, X_test, y_test, meta_train, meta_test = data

    # 2. Build sequences for LSTM and CNN
    #    Built from training data
    X_train_seq, y_train_seq = build_sequences(
        X_train, y_train, meta_train, seq_len=SEQ_LEN
    )
    #    Also build test sequences for Step 8 evaluation
    X_test_seq, y_test_seq = build_sequences(
        X_test, y_test, meta_test, seq_len=SEQ_LEN
    )

    # Save test sequences for Step 8
    np.save(os.path.join(OUTPUT_DIR, "X_test_seq.npy"), X_test_seq)
    np.save(os.path.join(OUTPUT_DIR, "y_test_seq.npy"), y_test_seq)
    print(f"\n  Test sequences saved for Step 8:")
    print(f"    X_test_seq : {X_test_seq.shape}")
    print(f"    y_test_seq : {y_test_seq.shape}")

    # 3. Train classical models
    classical_models = train_classical_models(X_train, y_train)

    # 4. Train deep learning models
    dl_models = train_deep_learning_models(
        X_train, y_train,
        X_train_seq, y_train_seq
    )

    # 5. Training summary
    all_models = list(classical_models.keys()) + list(dl_models.keys())
    print("\n" + "="*55)
    print("TRAINING COMPLETE — ALL 11 MODELS")
    print("="*55)
    print(f"\n  {'#':<4} {'Model':<30} {'Format':<12} {'Saved'}")
    print("  " + "-"*60)
    for i, name in enumerate(all_models, 1):
        fmt  = "Sequence" if name in ["10_CNN1D", "11_LSTM"] else "Flat"
        ext  = ".pt" if i >= 8 else ".pkl"
        path = os.path.join(MODELS_DIR, f"{name}{ext}")
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {i:<4} {name:<30} {fmt:<12} {exists} {path}")

    # Save training log
    log_path = os.path.join(OUTPUT_DIR, "step7_training_log.txt")
    with open(log_path, "w") as f:
        f.write("Step 7 — Training Log\n")
        f.write("="*40 + "\n")
        f.write(f"X_train shape : {X_train.shape}\n")
        f.write(f"y_train EQ %  : {y_train.mean()*100:.1f}%\n")
        f.write(f"X_test shape  : {X_test.shape}\n")
        f.write(f"y_test EQ %   : {y_test.mean()*100:.1f}%\n")
        f.write(f"Seq length    : {SEQ_LEN}\n")
        f.write(f"DL epochs     : {EPOCHS}\n")
        f.write(f"Models trained: {len(all_models)}\n")
        for name in all_models:
            f.write(f"  {name}\n")
    print(f"\n  Log saved: {log_path}")

    print("\n✅ Step 7 complete.")
    print("   Next → Step 8: Evaluate all models")


if __name__ == "__main__":
    main()