import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier,
                                    GradientBoostingClassifier)
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.naive_bayes   import GaussianNB
from sklearn.metrics       import (precision_score, recall_score,
                                    f1_score, roc_auc_score,
                                    confusion_matrix, roc_curve)

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
FLAT_FEATURES = SEQ_LEN * N_FEATURES  # 168
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 0.001
PATIENCE      = 20

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

FEATURE_COLS = ["Amp_Mean", "Amp_Std",
                "Phase_Mean", "Phase_Std"]

MODEL_NAMES = {
    "01_LogisticRegression" : "Logistic Regression",
    "02_DecisionTree"       : "Decision Tree",
    "03_RandomForest"       : "Random Forest",
    "04_GradientBoosting"   : "Gradient Boosting",
    "05_SVM"                : "SVM",
    "06_KNN"                : "KNN",
    "07_NaiveBayes"         : "Naïve Bayes",
    "08_SimpleNN"           : "Simple NN",
    "09_DeepNN"             : "Deep NN",
    "10_CNN1D"              : "1D-CNN",
    "11_LSTM"               : "LSTM",
}


#Model Definitions

class SimpleNN(nn.Module):
    def __init__(self, input_size=FLAT_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(1)


class DeepNN(nn.Module):
    def __init__(self, input_size=FLAT_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(1)


class CNN1D(nn.Module):
    def __init__(self, n_features=N_FEATURES,
                 seq_len=SEQ_LEN):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, 32,
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
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.classifier(
            self.conv_block(x)).squeeze(1)


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
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(
            out[:, -1, :]).squeeze(1)


# ── Load Data ─────────────────────────────────────────────────

def load_data():
    print("Loading sequence data...")

    files = ["X_train_seq.npy", "y_train_seq.npy",
             "X_val_seq.npy",   "y_val_seq.npy",
             "X_test_seq.npy",  "y_test_seq.npy",
             "X_train_flat.npy","X_val_flat.npy",
             "X_test_flat.npy"]

    for f in files:
        if not os.path.exists(
                os.path.join(OUTPUT_DIR, f)):
            print(f"  ERROR: {f} not found.")
            print("  Run build_sequences.py first.")
            return None

    d = {}
    for f in files:
        key    = f.replace(".npy", "")
        d[key] = np.load(
            os.path.join(OUTPUT_DIR, f))
        print(f"  {f:<22} : {d[key].shape}")

    return d


# PyTorch Training

def train_pytorch(model, X_train, y_train,
                  X_val, y_val, name):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu")
    model  = model.to(device)

    n_pos      = int(y_train.sum())
    n_neg      = len(y_train) - n_pos
    pos_weight = torch.tensor(
        [n_neg / n_pos],
        dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val,   dtype=torch.float32)\
               .to(device)
    y_v = torch.tensor(y_val,   dtype=torch.float32)\
               .to(device)

    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=BATCH_SIZE, shuffle=True)

    train_hist = []
    val_hist   = []
    best_val   = float('inf')
    best_epoch = 0
    no_improve = 0

    model.train()
    print(f"    Training on {device}...")

    for epoch in range(EPOCHS):
        total = 0
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        train_hist.append(avg)

        model.eval()
        with torch.no_grad():
            vl = criterion(
                model(X_v), y_v).item()
        val_hist.append(vl)
        model.train()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:>3}/{EPOCHS} "
                  f"Train={avg:.4f} Val={vl:.4f}")

        if vl < best_val:
            best_val   = vl
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(),
                os.path.join(MODELS_DIR,
                             f"{name}_best.pt"))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stop epoch {epoch+1}"
                      f" — best epoch {best_epoch}"
                      f" (val={best_val:.4f})")
                break

    best_path = os.path.join(
        MODELS_DIR, f"{name}_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(
            best_path, map_location="cpu"))
        print(f"    Best model restored "
              f"from epoch {best_epoch}")

    model.eval()
    return model, train_hist, val_hist


# Predict

def predict_classical(model, X):
    return (model.predict(X),
            model.predict_proba(X)[:, 1])

def predict_pytorch(model, X):
    with torch.no_grad():
        logits = model(torch.tensor(
            X, dtype=torch.float32)).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return preds, probs


# Metrics

def metrics(y_true, y_pred, y_prob, name):
    return {
        "model"    : name,
        "precision": round(precision_score(
            y_true, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(
            y_true, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(
            y_true, y_pred, zero_division=0), 4),
        "auc"      : round(roc_auc_score(
            y_true, y_prob), 4),
        "cm"       : confusion_matrix(y_true, y_pred),
        "y_true"   : y_true,
        "y_prob"   : y_prob,
        "y_pred"   : y_pred,
    }


# Feature importance by day 

def plot_feature_importance(gb_model):
    print("\nGenerating feature importance by day...")

    feat_path = os.path.join(
        OUTPUT_DIR, "feature_names.txt")
    with open(feat_path) as f:
        feature_names = [l.strip() for l in f]

    importances = gb_model.feature_importances_
    imp_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importances
    })

    # Extract day number
    imp_df["day"] = imp_df["feature"].str.extract(
        r"Day-(\d+)").astype(int)
    imp_df["feat_type"] = imp_df["feature"].str.extract(
        r"_([^_]+)$")

    # Group by day
    day_imp = imp_df.groupby("day")[
        "importance"].sum().reset_index()
    day_imp = day_imp.sort_values(
        "day", ascending=False)
    day_imp["day_label"] = day_imp["day"].apply(
        lambda d: f"Day -{d}" if d > 0 else "Day 0\n(EQ day)")

    # Group by feature type
    feat_imp = imp_df.groupby("feat_type")[
        "importance"].sum().reset_index()
    feat_imp = feat_imp.sort_values(
        "importance", ascending=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Day importance
    ax = axes[0]
    colors = ["#C0392B" if d == 0 else "#2E6DA4"
              for d in day_imp["day"]]
    bars = ax.barh(day_imp["day_label"],
                   day_imp["importance"],
                   color=colors, alpha=0.85,
                   edgecolor="white")
    for bar, val in zip(bars, day_imp["importance"]):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}",
                va="center", fontsize=9,
                color="#2D3748")
    ax.set_xlabel("Feature importance", fontsize=11)
    ax.set_title(
        "GB Feature Importance by Day\n",
        fontsize=11,
        color="#010402", fontname= "Times New Roman")
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(1/7, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.5,
               label="Equal importance (1/7)")
    ax.legend(fontsize=8)

    # Feature type importance
    ax = axes[1]
    feat_colors = ["#C0392B", "#2E6DA4",
                   "#2E7D52", "#C8922A"]
    bars2 = ax.bar(feat_imp["feat_type"],
                   feat_imp["importance"],
                   color=feat_colors[:len(feat_imp)],
                   alpha=0.85, edgecolor="white",
                   width=0.5)
    for bar, val in zip(bars2, feat_imp["importance"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center", fontsize=9,
                color="#2D3748")
    ax.set_ylabel("Feature importance", fontsize=11)
    ax.set_title(
        "GB Feature Importance by Type\n"
        "Amplitude vs Phase",
        fontsize=11, 
        color="#070A04", fontname= "Times New Roman")
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Gradient Boosting: Feature Importance Analysis\n"
        ,
        fontsize=12, fontweight="bold",
        color="#1B3A5C", y=1.02, fontname = "Times New Roman")
    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR, "feature_importance_by_day.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

    print("\n  Feature importance by day:")
    print(f"  {'Day':<12} {'Importance':>12}")
    print("  " + "-"*26)
    for _, row in day_imp.iterrows():
        marker = " ← highest" \
            if row["importance"] == \
               day_imp["importance"].max() \
            else ""
        print(f"  Day -{int(row['day']):<8} "
              f"{row['importance']:>12.4f}{marker}")

    return day_imp


# Plot loss curves

def plot_loss_curves(histories):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
    
        "10_CNN1D"   : "#C0392B",
        "11_LSTM"    : "#C8922A",
    }
    for name, (train_h, val_h) in histories.items():
        if name not in ["10_CNN1D", "11_LSTM"]:
            continue
        lbl   = MODEL_NAMES[name]
        color = colors.get(name, "gray")
        ax.plot(range(1, len(train_h)+1), train_h,
                label=f"{lbl} train",
                color=color, linewidth=1.5,
                linestyle="-")
        ax.plot(range(1, len(val_h)+1), val_h,
                label=f"{lbl} val",
                color=color, linewidth=1.5,
                linestyle="--")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(
        "Training and Validation Loss Curves\n"
        "Solid = Training & Dashed = Validation",
        fontsize=12, fontweight="bold",
        color="#1B3A5C", fontname = "Times New Roman")
    ax.legend(fontsize=9, ncol=2)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        "loss_curves.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Loss curves: {path}")

# Plot results
def plot_results(results):
    names = [MODEL_NAMES[r["model"]]
             for r in results]
    prec  = [r["precision"] for r in results]
    rec   = [r["recall"]    for r in results]
    f1s   = [r["f1"]        for r in results]
    aucs  = [r["auc"]       for r in results]

    x = np.arange(len(results))
    w = 0.2

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("white")

    ax.bar(x-1.5*w, prec, w, label="Precision",
           color="#2E6DA4", alpha=0.85,
           edgecolor="white")
    ax.bar(x-0.5*w, rec,  w, label="Recall",
           color="#C0392B", alpha=0.85,
           edgecolor="white")
    ax.bar(x+0.5*w, f1s,  w, label="F1-Score",
           color="#2E7D52", alpha=0.85,
           edgecolor="white")
    ax.bar(x+1.5*w, aucs, w, label="AUC-ROC",
           color="#C8922A", alpha=0.85,
           edgecolor="white")

    ax.axvline(6.5, color="#7A92A8",
               linewidth=1.2, linestyle="--",
               alpha=0.7)
    ax.text(3.0, 1.02, "Classical models",
            ha="center", fontsize=9,
            color="#7A92A8")
    ax.text(8.5, 1.02, "Deep learning",
            ha="center", fontsize=9,
            color="#7A92A8")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35,
                       ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Evaluation Metrics — All 11 Models\n"
        ,
        fontsize=11, fontweight="bold",
        color="#1B3A5C", fontname= "Times New Roman")
    ax.legend(fontsize=10)
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        "evaluation_metrics.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Metrics plot: {path}")

    # ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    cmap = plt.cm.get_cmap("tab20", len(results))
    clean_names = [MODEL_NAMES[r["model"]]
                   for r in results]
    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(
            r["y_true"], r["y_prob"])
        ax.plot(fpr, tpr, color=cmap(i),
                linewidth=1.5,
                label=f"{clean_names[i]} "
                      f"(AUC={r['auc']:.3f})")
    ax.plot([0,1],[0,1],"k--",linewidth=1,
            alpha=0.5, label="Random (0.5)")
    ax.set_xlabel("False Positive Rate",
                  fontsize=11)
    ax.set_ylabel("True Positive Rate",
                  fontsize=11)
    ax.set_title(
        "ROC Curves — All 11 Models",
        fontsize=12, fontweight="bold",
        color="#1B3A5C")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_facecolor("#F7FAFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR,
                         "roc_curves.png")
    plt.savefig(path2, dpi=150,
                bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  ROC curves  : {path2}")

def plot_confusion_matrices(results):
    print("\nGenerating confusion matrices...")

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("white")

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.45, wspace=0.35)

    clean_names = [MODEL_NAMES[r["model"]]
                   for r in results]

    for i, r in enumerate(results):
        ax   = fig.add_subplot(gs[i // 4, i % 4])
        cm   = r["cm"]
        name = clean_names[i]

        cm_norm = cm.astype(float) / \
                  cm.sum(axis=1, keepdims=True)

        ax.imshow(cm_norm, interpolation="nearest",
                  cmap="Blues", vmin=0, vmax=1)

        labels = [["TN", "FP"], ["FN", "TP"]]
        for row in range(2):
            for col in range(2):
                count = cm[row, col]
                pct   = cm_norm[row, col]
                ax.text(col, row,
                        f"{labels[row][col]}\n"
                        f"{count}\n"
                        f"({pct:.1%})",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white"
                        if pct > 0.5 else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(
            ["Pred 0\n(Normal)",
             "Pred 1\n(Pre-EQ)"], fontsize=8)
        ax.set_yticklabels(
            ["True 0\n(Normal)",
             "True 1\n(Pre-EQ)"], fontsize=8)
        ax.set_title(
            f"{name}\n"
            f"F1={r['f1']:.3f} "
            f"AUC={r['auc']:.3f}",
            fontsize=9, fontweight="bold",
            color="#1B3A5C")

    # Hide last empty subplot
    if len(results) < 12:
        ax_empty = fig.add_subplot(gs[2, 3])
        ax_empty.axis("off")

    fig.suptitle(
        "Confusion Matrices — All 11 Models\n",
        fontsize=13, fontweight="bold",
        color="#80242D", y=1.01, fontname = "Times New Roman")

    path = os.path.join(OUTPUT_DIR,
                        "confusion_matrices.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Confusion matrices: {path}")
def plot_best_model_cm(results):
    """Detailed confusion matrix for best model by F1."""
    print("\nGenerating best model confusion matrix...")

    best = max(results, key=lambda r: r["f1"])
    name = MODEL_NAMES[best["model"]]
    cm   = best["cm"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # Raw counts
    ax = axes[0]
    im = ax.imshow(cm, cmap="Greens",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{labels[i][j]}\n{cm[i,j]:,}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(
        ["Predicted Normal", "Predicted Pre-EQ"],
        fontsize=10)
    ax.set_yticklabels(
        ["Actual Normal", "Actual Pre-EQ"],
        fontsize=10)
    ax.set_title(f"{name}  (Counts)",
                 fontsize=11, fontweight="bold",
                 color="#1B3A5C", fontname = "Times New Roman")

    # Normalised
    ax = axes[1]
    cm_norm = cm.astype(float) / \
              cm.sum(axis=1, keepdims=True)
    im2 = ax.imshow(cm_norm, cmap="Blues",
                    interpolation="nearest",
                    vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax)
    lab2 = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{lab2[i][j]}\n"
                    f"{cm[i,j]}\n"
                    f"({cm_norm[i,j]:.1%})",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white"
                    if cm_norm[i,j] > 0.5
                    else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(
        ["Predicted Normal", "Predicted Pre-EQ"],
        fontsize=10)
    ax.set_yticklabels(
        ["Actual Normal", "Actual Pre-EQ"],
        fontsize=10)
    ax.set_title(
        f"{name}  (Normalised)\n"
        f"Precision={best['precision']:.4f}  "
        f"Recall={best['recall']:.4f}  "
        f"F1={best['f1']:.4f}  "
        f"AUC={best['auc']:.4f}",
        fontsize=10, fontweight="bold",
        color="#1B3A5C", fontname = "Times New Roman")

    plt.suptitle(
        f"Best Model: {name}\n"
        f"Test set: 228 independent sequences",
        fontsize=13, fontweight="bold",
        color="#1B3A5C", y=1.02)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        "best_model_cm.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Best model : {name}")
    print(f"  F1={best['f1']:.4f}  "
          f"AUC={best['auc']:.4f}  "
          f"Recall={best['recall']:.4f}")
# Main

def main():
    print("=" * 55)
    print("TRAIN AND EVALUATE — NEW PIPELINE")
    print("=" * 55)

    data = load_data()
    if data is None:
        return

    X_tr_seq  = data["X_train_seq"]
    y_tr_seq  = data["y_train_seq"]
    X_v_seq   = data["X_val_seq"]
    y_v_seq   = data["y_val_seq"]
    X_te_seq  = data["X_test_seq"]
    y_te_seq  = data["y_test_seq"]
    X_tr_flat = data["X_train_flat"]
    X_v_flat  = data["X_val_flat"]
    X_te_flat = data["X_test_flat"]

    results   = []
    histories = {}

    # Classical models
    print("\n" + "="*55)
    print("CLASSICAL MODELS (7) — 168 features")
    print("="*55)

    classical = {
        "01_LogisticRegression":
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=2000),
        "02_DecisionTree":
            DecisionTreeClassifier(
                random_state=RANDOM_STATE),
        "03_RandomForest":
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1),
        "04_GradientBoosting":
            GradientBoostingClassifier(
                random_state=RANDOM_STATE),
        "05_SVM":
            SVC(probability=True,
                random_state=RANDOM_STATE),
        "06_KNN":
            KNeighborsClassifier(n_jobs=-1),
        "07_NaiveBayes":
            GaussianNB(),
    }

    gb_model = None
    for name, model in classical.items():
        print(f"\n  Training {name}...")
        start = time.time()
        model.fit(X_tr_flat, y_tr_seq)
        elapsed = time.time() - start
        path = os.path.join(
            MODELS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        yp, yprob = predict_classical(
            model, X_te_flat)
        r = metrics(y_te_seq, yp, yprob, name)
        results.append(r)
        print(f"  ✓ {elapsed:.1f}s — "
              f"F1={r['f1']:.4f} "
              f"AUC={r['auc']:.4f}")
        if name == "04_GradientBoosting":
            gb_model = model

    # SimpleNN and DeepNN
    print("\n" + "="*55)
    print("FLAT NEURAL NETWORKS — 168 features")
    print("="*55)

    flat_dl = [
        ("08_SimpleNN",
         SimpleNN(FLAT_FEATURES)),
        ("09_DeepNN",
         DeepNN(FLAT_FEATURES)),
    ]

    for name, model in flat_dl:
        print(f"\n  Training {name}...")
        print(f"    Train: {X_tr_flat.shape} "
              f"Val: {X_v_flat.shape}")
        start = time.time()
        trained, th, vh = train_pytorch(
            model, X_tr_flat, y_tr_seq,
            X_v_flat, y_v_seq, name)
        elapsed = time.time() - start
        torch.save(trained.state_dict(),
            os.path.join(MODELS_DIR,
                         f"{name}.pt"))
        yp, yprob = predict_pytorch(
            trained, X_te_flat)
        r = metrics(y_te_seq, yp, yprob, name)
        results.append(r)
        histories[name] = (th, vh)
        print(f"  ✓ {elapsed:.1f}s — "
              f"F1={r['f1']:.4f} "
              f"AUC={r['auc']:.4f}")

    # CNN1D and LSTM
    print("\n" + "="*55)
    print("SEQUENCE MODELS — (42, 4)")
    print("="*55)

    seq_dl = [
        ("10_CNN1D",  CNN1D()),
        ("11_LSTM",   LSTMModel()),
    ]

    for name, model in seq_dl:
        print(f"\n  Training {name}...")
        print(f"    Train: {X_tr_seq.shape} "
              f"Val: {X_v_seq.shape}")
        start = time.time()
        trained, th, vh = train_pytorch(
            model, X_tr_seq, y_tr_seq,
            X_v_seq, y_v_seq, name)
        elapsed = time.time() - start
        torch.save(trained.state_dict(),
            os.path.join(MODELS_DIR,
                         f"{name}.pt"))
        yp, yprob = predict_pytorch(
            trained, X_te_seq)
        r = metrics(y_te_seq, yp, yprob, name)
        results.append(r)
        histories[name] = (th, vh)
        print(f"  ✓ {elapsed:.1f}s — "
              f"F1={r['f1']:.4f} "
              f"AUC={r['auc']:.4f}")

    # Feature importance
    if gb_model is not None:
        plot_feature_importance(gb_model)

    # Loss curves
    plot_loss_curves(histories)

    # Results table
    print("\n" + "="*65)
    print("FINAL RESULTS (sorted by F1)")
    print("="*65)
    print(f"\n  {'Model':<30} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} "
          f"{'AUC':>10}")
    print("  " + "-"*72)

    results_sorted = sorted(
        results, key=lambda r: r["f1"],
        reverse=True)

    for r in results_sorted:
        name = MODEL_NAMES[r["model"]]
        print(f"  {name:<30} "
              f"{r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} "
              f"{r['f1']:>10.4f} "
              f"{r['auc']:>10.4f}")

    # Save results CSV
    rows = []
    for r in results_sorted:
        rows.append({
            "Model"    : MODEL_NAMES[r["model"]],
            "Precision": r["precision"],
            "Recall"   : r["recall"],
            "F1"       : r["f1"],
            "AUC"      : r["auc"],
        })
    df_results = pd.DataFrame(rows)
    csv_path   = os.path.join(
        OUTPUT_DIR, "final_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")

    # Plot
# Plot
    plot_results(results)
    plot_confusion_matrices(results)

    plot_best_model_cm(results)
    # Best model summary
    best_f1  = max(results,
                   key=lambda r: r["f1"])
    best_auc = max(results,
                   key=lambda r: r["auc"])
    best_rec = max(results,
                   key=lambda r: r["recall"])

    print("\n" + "="*55)
    print("BEST MODEL SUMMARY")
    print("="*55)
    print(f"  Best F1   : "
          f"{MODEL_NAMES[best_f1['model']]} "
          f"(F1={best_f1['f1']:.4f})")
    print(f"  Best AUC  : "
          f"{MODEL_NAMES[best_auc['model']]} "
          f"(AUC={best_auc['auc']:.4f})")
    print(f"  Best Recall: "
          f"{MODEL_NAMES[best_rec['model']]} "
          f"(Recall={best_rec['recall']:.4f})")

    print("\n Train and evaluate complete.")


if __name__ == "__main__":
    main()