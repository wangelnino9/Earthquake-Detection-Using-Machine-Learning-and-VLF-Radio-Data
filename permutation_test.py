import numpy as np
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import (RandomForestClassifier,
                                   GradientBoostingClassifier)
from sklearn.metrics       import (f1_score, roc_auc_score,
                                   precision_score,
                                   recall_score)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR   = "outputs"
RANDOM_STATE = 42
SEQ_LEN      = 42
N_FEATURES   = 4
FLAT_FEATURES = SEQ_LEN * N_FEATURES
BATCH_SIZE   = 32
EPOCHS       = 100
PATIENCE     = 20
LEARNING_RATE = 0.001

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MODEL_NAMES = {
    "01_LogisticRegression" : "Logistic Regression",
    "03_RandomForest"       : "Random Forest",
    "04_GradientBoosting"   : "Gradient Boosting",
    "10_CNN1D"              : "1D-CNN",
    "11_LSTM"               : "LSTM",
}


#Model definitions

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(N_FEATURES, 32,
                      kernel_size=6, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64,
                      kernel_size=6, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.classifier(
            self.conv_block(
                x.permute(0, 2, 1))).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            N_FEATURES, 64, 2,
            batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(
            out[:, -1, :]).squeeze(1)


# Train PyTorch with random labels

def train_pytorch_random(model, X_train, y_train,
                          X_val, y_val):
    device    = torch.device("cpu")
    model     = model.to(device)
    n_pos     = int(y_train.sum())
    n_neg     = len(y_train) - n_pos

    if n_pos == 0 or n_neg == 0:
        pos_weight = torch.tensor([1.0])
    else:
        pos_weight = torch.tensor(
            [n_neg / n_pos], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val,   dtype=torch.float32)
    y_v = torch.tensor(y_val,   dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=BATCH_SIZE, shuffle=True)

    best_val   = float('inf')
    best_epoch = 0
    no_improve = 0
    best_state = None

    model.train()
    for epoch in range(EPOCHS):
        for Xb, yb in loader:
            optimizer.zero_grad()
            nn.BCEWithLogitsLoss()(
                model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_v), y_v).item()
        model.train()

        if vl < best_val:
            best_val   = vl
            best_epoch = epoch + 1
            best_state = {k: v.clone()
                for k, v in
                model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_pytorch(model, X):
    with torch.no_grad():
        logits = model(torch.tensor(
            X, dtype=torch.float32)).numpy()
    probs = 1 / (1 + np.exp(-logits))
    return (probs >= 0.5).astype(int), probs


# Main

def main():
    print("=" * 60)
    print("PERMUTATION TEST — RANDOMISED LABELS")
    print("=" * 60)
    print("Keeps sequences unchanged")
    print("Randomises training labels only")
    print("Tests are still evaluated on real labels")

    # Load data
    X_tr_seq  = np.load(os.path.join(
        OUTPUT_DIR, "X_train_seq.npy"))
    y_tr_seq  = np.load(os.path.join(
        OUTPUT_DIR, "y_train_seq.npy"))
    X_v_seq   = np.load(os.path.join(
        OUTPUT_DIR, "X_val_seq.npy"))
    y_v_seq   = np.load(os.path.join(
        OUTPUT_DIR, "y_val_seq.npy"))
    X_te_seq  = np.load(os.path.join(
        OUTPUT_DIR, "X_test_seq.npy"))
    y_te_seq  = np.load(os.path.join(
        OUTPUT_DIR, "y_test_seq.npy"))
    X_tr_flat = np.load(os.path.join(
        OUTPUT_DIR, "X_train_flat.npy"))
    X_v_flat  = np.load(os.path.join(
        OUTPUT_DIR, "X_val_flat.npy"))
    X_te_flat = np.load(os.path.join(
        OUTPUT_DIR, "X_test_flat.npy"))

    print(f"\n  Train sequences : {X_tr_seq.shape}")
    print(f"  Test  sequences : {X_te_seq.shape}")

    # Randomise training labels
    np.random.seed(RANDOM_STATE)
    y_tr_random   = np.random.permutation(y_tr_seq)
    y_v_random    = np.random.permutation(y_v_seq)

    print(f"\n  Real labels    — "
          f"EQ={y_tr_seq.sum()}, "
          f"no-EQ={(y_tr_seq==0).sum()}")
    print(f"  Random labels  — "
          f"EQ={y_tr_random.sum()}, "
          f"no-EQ={(y_tr_random==0).sum()}")
    print(f"  (test labels unchanged — "
          f"EQ={y_te_seq.sum()}, "
          f"no-EQ={(y_te_seq==0).sum()})")

    #Real label results
    real_results = {
        "Logistic Regression" : (0.6043, 0.6228, 0.6043, 0.6483),
        "Random Forest"       : (0.7500, 0.6316, 0.6857, 0.7766),
        "Gradient Boosting"   : (0.7500, 0.6842, 0.7156, 0.7738),
        "1D-CNN"              : (0.6923, 0.7105, 0.7013, 0.7499),
        "LSTM"                : (0.6226, 0.5789, 0.6000, 0.6661),
    }

    #Train with random labels
    print("\n" + "="*60)
    print("TRAINING WITH RANDOMISED LABELS")
    print("="*60)

    random_results = {}

    # Logistic Regression
    print("\n  Training Logistic Regression...")
    lr = LogisticRegression(
        random_state=RANDOM_STATE, max_iter=2000)
    lr.fit(X_tr_flat, y_tr_random)
    yp   = lr.predict(X_te_flat)
    yprob = lr.predict_proba(X_te_flat)[:, 1]
    random_results["Logistic Regression"] = (
        precision_score(y_te_seq, yp, zero_division=0),
        recall_score(y_te_seq, yp, zero_division=0),
        f1_score(y_te_seq, yp, zero_division=0),
        roc_auc_score(y_te_seq, yprob)
    )
    print(f"  F1={random_results['Logistic Regression'][2]:.4f} "
          f"AUC={random_results['Logistic Regression'][3]:.4f}")

    # Random Forest
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_flat, y_tr_random)
    yp    = rf.predict(X_te_flat)
    yprob = rf.predict_proba(X_te_flat)[:, 1]
    random_results["Random Forest"] = (
        precision_score(y_te_seq, yp, zero_division=0),
        recall_score(y_te_seq, yp, zero_division=0),
        f1_score(y_te_seq, yp, zero_division=0),
        roc_auc_score(y_te_seq, yprob)
    )
    print(f"  F1={random_results['Random Forest'][2]:.4f} "
          f"AUC={random_results['Random Forest'][3]:.4f}")

    # Gradient Boosting
    print("\n  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        random_state=RANDOM_STATE)
    gb.fit(X_tr_flat, y_tr_random)
    yp    = gb.predict(X_te_flat)
    yprob = gb.predict_proba(X_te_flat)[:, 1]
    random_results["Gradient Boosting"] = (
        precision_score(y_te_seq, yp, zero_division=0),
        recall_score(y_te_seq, yp, zero_division=0),
        f1_score(y_te_seq, yp, zero_division=0),
        roc_auc_score(y_te_seq, yprob)
    )
    print(f"  F1={random_results['Gradient Boosting'][2]:.4f} "
          f"AUC={random_results['Gradient Boosting'][3]:.4f}")

    # CNN1D
    print("\n  Training 1D-CNN...")
    cnn = CNN1D()
    cnn = train_pytorch_random(
        cnn, X_tr_seq, y_tr_random,
        X_v_seq, y_v_random)
    yp, yprob = predict_pytorch(cnn, X_te_seq)
    random_results["1D-CNN"] = (
        precision_score(y_te_seq, yp, zero_division=0),
        recall_score(y_te_seq, yp, zero_division=0),
        f1_score(y_te_seq, yp, zero_division=0),
        roc_auc_score(y_te_seq, yprob)
    )
    print(f"  F1={random_results['1D-CNN'][2]:.4f} "
          f"AUC={random_results['1D-CNN'][3]:.4f}")

    # LSTM
    print("\n  Training LSTM...")
    lstm = LSTMModel()
    lstm = train_pytorch_random(
        lstm, X_tr_seq, y_tr_random,
        X_v_seq, y_v_random)
    yp, yprob = predict_pytorch(lstm, X_te_seq)
    random_results["LSTM"] = (
        precision_score(y_te_seq, yp, zero_division=0),
        recall_score(y_te_seq, yp, zero_division=0),
        f1_score(y_te_seq, yp, zero_division=0),
        roc_auc_score(y_te_seq, yprob)
    )
    print(f" F1={random_results['LSTM'][2]:.4f} "
          f"AUC={random_results['LSTM'][3]:.4f}")

    #Comparison table
    print("\n" + "="*70)
    print("COMPARISON — REAL LABELS vs RANDOM LABELS")
    print("="*70)
    print(f"\n  {'Model':<22} "
          f"{'Real F1':>8} {'Rand F1':>8} "
          f"{'Real AUC':>9} {'Rand AUC':>9} "
          f"{'Drop F1':>8}")
    print("  " + "-"*66)

    rows = []
    for model_name in real_results:
        r_prec, r_rec, r_f1, r_auc = \
            real_results[model_name]
        rnd_prec, rnd_rec, rnd_f1, rnd_auc = \
            random_results[model_name]
        drop_f1 = r_f1 - rnd_f1
        print(f"  {model_name:<22} "
              f"{r_f1:>8.4f} {rnd_f1:>8.4f} "
              f"{r_auc:>9.4f} {rnd_auc:>9.4f} "
              f"{drop_f1:>8.4f}")
        rows.append({
            "Model"       : model_name,
            "Real_F1"     : r_f1,
            "Random_F1"   : rnd_f1,
            "Real_AUC"    : r_auc,
            "Random_AUC"  : rnd_auc,
            "F1_Drop"     : round(drop_f1, 4),
        })

    # Save CSV
    df = pd.DataFrame(rows)
    path = os.path.join(
        OUTPUT_DIR, "permutation_test.csv")
    df.to_csv(path, index=False)
    print(f"\n  Results saved: {path}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    avg_real   = np.mean([r[2] for r in
                          real_results.values()])
    avg_random = np.mean([r[2] for r in
                          random_results.values()])
    print(f"\n  Average F1 with real labels   : "
          f"{avg_real:.4f}")
    print(f"  Average F1 with random labels : "
          f"{avg_random:.4f}")
    print(f"  Average drop                  : "
          f"{avg_real - avg_random:.4f}")

    if avg_random < 0.55:
        print(f"\n  CONFIRMED: Models drop to near-random")
        print(f"    with randomised labels.")
        print(f"    Real results are driven by genuine")
        print(f"    pre-seismic VLF signal.")
    else:
        print(f"\n  WARNING: Models still perform above")
        print(f"    random with shuffled labels.")
        print(f"    Investigate data structure.")

    print("\nPermutation test complete.")


if __name__ == "__main__":
    main()