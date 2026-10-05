import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score
)

OUTPUT_DIR = "outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

N_FEATURES = 4
SEQ_LEN    = 42

# Colour palette
NAVY   = "#1B3A5C"
BLUE   = "#2E6DA4"
GREEN  = "#2E7D52"
RED    = "#C0392B"
GOLD   = "#C8922A"
STEEL  = "#5B8DB8"
MUTED  = "#7A92A8"
LIGHT  = "#F7FAFD"
WHITE  = "#FFFFFF"

# Clean model names for plots
MODEL_NAMES = {
    "01_LogisticRegression": "Logistic Regression",
    "02_DecisionTree":       "Decision Tree",
    "03_RandomForest":       "Random Forest",
    "04_GradientBoosting":   "Gradient Boosting",
    "05_SVM":                "SVM",
    "06_KNN":                "KNN",
    "07_NaiveBayes":         "Naïve Bayes",
    "08_SimpleNN":           "Simple NN",
    "09_DeepNN":             "Deep NN",
    "10_CNN1D":              "1D-CNN",
    "11_LSTM":               "LSTM",
}

# Model type for colour coding
MODEL_TYPE = {
    "01_LogisticRegression": "classical",
    "02_DecisionTree":       "classical",
    "03_RandomForest":       "classical",
    "04_GradientBoosting":   "classical",
    "05_SVM":                "classical",
    "06_KNN":                "classical",
    "07_NaiveBayes":         "classical",
    "08_SimpleNN":           "deep",
    "09_DeepNN":             "deep",
    "10_CNN1D":              "deep",
    "11_LSTM":               "deep",
}


# PyTorch Model Definitions

class SimpleNN(nn.Module):
    def __init__(self, input_size=N_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        return self.network(x).squeeze(1)

class DeepNN(nn.Module):
    def __init__(self, input_size=N_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1))
    def forward(self, x):
        return self.network(x).squeeze(1)

class CNN1D(nn.Module):
    def __init__(self, n_features=N_FEATURES, seq_len=SEQ_LEN):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=6, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=6, padding=3),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.classifier(self.conv_block(x)).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self, input_size=N_FEATURES,
                 hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True,
                            dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(1)


# Load Data

def load_test_data():
    print("Loading test data...")
    X_test     = pd.read_csv(
        os.path.join(OUTPUT_DIR, "X_test.csv")).values
    y_test     = pd.read_csv(
        os.path.join(OUTPUT_DIR, "y_test.csv")).values.ravel()
    X_test_seq = np.load(
        os.path.join(OUTPUT_DIR, "X_test_seq.npy"))
    y_test_seq = np.load(
        os.path.join(OUTPUT_DIR, "y_test_seq.npy"))
    print(f"  Flat  : {X_test.shape} — "
          f"EQ={y_test.sum()}, no-EQ={(y_test==0).sum()}")
    print(f"  Seq   : {X_test_seq.shape} — "
          f"EQ={y_test_seq.sum()}, no-EQ={(y_test_seq==0).sum()}")
    return X_test, y_test, X_test_seq, y_test_seq


def load_classical(name):
    with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)

def load_pytorch(name, cls):
    m = cls()
    m.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, f"{name}.pt"),
        map_location="cpu"))
    m.eval()
    return m

def predict_classical(model, X):
    return model.predict(X), model.predict_proba(X)[:, 1]

def predict_pytorch(model, X, threshold=0.5):
    with torch.no_grad():
        logits = model(
            torch.tensor(X, dtype=torch.float32)).numpy()
        probs  = 1 / (1 + np.exp(-logits))  # manual sigmoid
    return (probs >= threshold).astype(int), probs


# Evaluate All Models

def evaluate_all(X_test, y_test, X_seq, y_seq):
    print("\nEvaluating all 11 models...")
    results = []

    classical = [
        "01_LogisticRegression", "02_DecisionTree",
        "03_RandomForest",       "04_GradientBoosting",
        "05_SVM",                "06_KNN",
        "07_NaiveBayes",
    ]
    for name in classical:
        m = load_classical(name)
        yp, yprob = predict_classical(m, X_test)

        # Lower threshold for GB to improve recall
        if name == "04_GradientBoosting":
            yp = (yprob >= 0.45).astype(int)

        results.append({
            "model":     name,
            "y_true":    y_test,
            "y_pred":    yp,
            "y_prob":    yprob,
            "precision": round(precision_score(y_test, yp, zero_division=0), 4),
            "recall":    round(recall_score(y_test, yp, zero_division=0), 4),
            "f1":        round(f1_score(y_test, yp, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_test, yprob), 4),
            "cm":        confusion_matrix(y_test, yp),
        })

    dl_flat = [("08_SimpleNN", SimpleNN), ("09_DeepNN", DeepNN)]
    for name, cls in dl_flat:
        m = load_pytorch(name, cls)
        yp, yprob = predict_pytorch(m, X_test)
        results.append({
            "model":     name,
            "y_true":    y_test,
            "y_pred":    yp,
            "y_prob":    yprob,
            "precision": round(precision_score(y_test, yp, zero_division=0), 4),
            "recall":    round(recall_score(y_test, yp, zero_division=0), 4),
            "f1":        round(f1_score(y_test, yp, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_test, yprob), 4),
            "cm":        confusion_matrix(y_test, yp),
        })

    dl_seq = [("10_CNN1D", CNN1D), ("11_LSTM", LSTMModel)]
    for name, cls in dl_seq:
        m = load_pytorch(name, cls)
        yp, yprob = predict_pytorch(m, X_seq)
        results.append({
            "model":     name,
            "y_true":    y_seq,
            "y_pred":    yp,
            "y_prob":    yprob,
            "precision": round(precision_score(y_seq, yp, zero_division=0), 4),
            "recall":    round(recall_score(y_seq, yp, zero_division=0), 4),
            "f1":        round(f1_score(y_seq, yp, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_seq, yprob), 4),
            "cm":        confusion_matrix(y_seq, yp),
        })

    return results


#  Clean Results Table

def save_results_table(results):
    print("\nBuilding results table...")

    rows = []
    for r in results:
        rows.append({
            "Model":     MODEL_NAMES[r["model"]],
            "Type":      "Classical" if MODEL_TYPE[r["model"]] == "classical"
                         else "Deep Learning",
            "Precision": r["precision"],
            "Recall":    r["recall"],
            "F1-Score":  r["f1"],
            "AUC-ROC":   r["auc"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    df.index += 1

    print(df.to_string())

    path = os.path.join(OUTPUT_DIR, "results_table.csv")
    df.to_csv(path)
    print(f"\n  Table saved: {path}")
    return df, results


# Bar Chart

def plot_bar_chart(results):
    print("\nGenerating bar chart...")

    names   = [MODEL_NAMES[r["model"]] for r in results]
    prec    = [r["precision"] for r in results]
    rec     = [r["recall"]    for r in results]
    f1      = [r["f1"]        for r in results]
    auc     = [r["auc"]       for r in results]

    x  = np.arange(len(results))
    w  = 0.2
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(WHITE)

    ax.bar(x - 1.5*w, prec, w, label="Precision",
           color=BLUE,  alpha=0.85, edgecolor=WHITE)
    ax.bar(x - 0.5*w, rec,  w, label="Recall",
           color=RED,   alpha=0.85, edgecolor=WHITE)
    ax.bar(x + 0.5*w, f1,   w, label="F1-Score",
           color=GREEN, alpha=0.85, edgecolor=WHITE)
    ax.bar(x + 1.5*w, auc,  w, label="AUC-ROC",
           color=GOLD,  alpha=0.85, edgecolor=WHITE)

    # Divider between classical and deep learning
    ax.axvline(6.5, color=MUTED, linewidth=1.2,
               linestyle="--", alpha=0.7)
    ax.text(3.0, 1.02, "Classical Models",
            ha="center", fontsize=9, color=MUTED)
    ax.text(8.5, 1.02, "Deep Learning",
            ha="center", fontsize=9, color=MUTED)

    ax.axhline(0.5, color="gray", linewidth=0.8,
               linestyle=":", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35,
                       ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Evaluation Metrics — All 11 Models\n"
        "JJI VLF Dataset | Path-level labelling | "
        "7-day pre-earthquake window | 4-hour windows",
        fontsize=11, fontweight="bold", color=NAVY
    )
    ax.legend(fontsize=10, loc="upper right",
              framealpha=0.9)
    ax.set_facecolor(LIGHT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.patch.set_facecolor(WHITE)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bar_chart.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {path}")


# ROC Curves

def plot_roc_curves(results):
    print("\nGenerating ROC curves...")

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(WHITE)

    # Separate colours for classical vs deep learning
    classical_colors = plt.cm.Blues(
        np.linspace(0.4, 0.9, 7))
    deep_colors = plt.cm.Reds(
        np.linspace(0.4, 0.9, 4))

    c_idx = 0
    d_idx = 0

    for r in results:
        name  = MODEL_NAMES[r["model"]]
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        auc   = r["auc"]

        if MODEL_TYPE[r["model"]] == "classical":
            color = classical_colors[c_idx]
            c_idx += 1
            lw    = 1.2
            ls    = "-"
        else:
            color = deep_colors[d_idx]
            d_idx += 1
            lw    = 2.0
            ls    = "--"

        ax.plot(fpr, tpr, color=color,
                linewidth=lw, linestyle=ls,
                label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:",
            linewidth=1, alpha=0.5,
            label="Random (AUC=0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "ROC Curves — All 11 Models\n"
        "Solid = Classical  |  Dashed = Deep Learning",
        fontsize=12, fontweight="bold", color=NAVY, fontfamily="Times New Roman"
    )
    ax.legend(fontsize=8, loc="lower right",
              framealpha=0.9, ncol=1)
    ax.set_facecolor(LIGHT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor(WHITE)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {path}")


# Best Model Confusion Matrix

def plot_confusion_matrix(results):
    print("\nGenerating confusion matrix for best model...")

    # Find best model by F1
    best = max(results, key=lambda r: r["f1"])
    name = MODEL_NAMES[best["model"]]
    cm   = best["cm"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(WHITE)

    # Plot 1 — raw counts
    ax = axes[0]
    im = ax.imshow(cm, cmap="Greens",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="black" if cm[i,j] > cm.max()/2
                    else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNo EQ", "Predicted\nEQ"],
                       fontsize=10)
    ax.set_yticklabels(["Actual\nNo EQ", "Actual\nEQ"],
                       fontsize=10)
    ax.set_title(f"{name}\nConfusion Matrix (Counts)",
                 fontsize=11, fontweight="bold", color=NAVY, fontfamily= "Times New Roman")

    # Plot 2 — normalised
    ax = axes[1]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im2 = ax.imshow(cm_norm, cmap="Blues",
                    interpolation="nearest",
                    vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax)

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{labels[i][j]}\n{cm[i,j]:,}\n"
                    f"({cm_norm[i,j]:.1%})",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm_norm[i,j] > 0.5
                    else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNo EQ", "Predicted\nEQ"],
                       fontsize=10)
    ax.set_yticklabels(["Actual\nNo EQ", "Actual\nEQ"],
                       fontsize=10)
    ax.set_title(f"{name}\nConfusion Matrix (Normalised)\n"
                 f"Precision={best['precision']:.4f}  "
                 f"Recall={best['recall']:.4f}  "
                 f"F1={best['f1']:.4f}  "
                 f"AUC={best['auc']:.4f}",
                 fontsize=10, fontweight="bold", color=NAVY,fontfamily = "Times New Roman")

    plt.suptitle(
        f"Best Model: {name}",
        fontsize=13, fontweight="bold",
        color=NAVY, y=1.02, fontfamily = "Times New Roman",
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        "confusion_matrix.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Best model: {name} "
          f"(F1={best['f1']:.4f}, AUC={best['auc']:.4f})")


# Summary Figure 

def plot_summary(results):
  
    print("\nGenerating summary figure...")

    names  = [MODEL_NAMES[r["model"]] for r in results]
    f1s    = [r["f1"]  for r in results]
    aucs   = [r["auc"] for r in results]
    types  = [MODEL_TYPE[r["model"]] for r in results]
    colors = [BLUE if t == "classical" else RED
              for t in types]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(WHITE)

    for ax, vals, metric in zip(
            axes, [f1s, aucs], ["F1-Score", "AUC-ROC"]):

        bars = ax.barh(names, vals,
                       color=colors, alpha=0.85,
                       edgecolor=WHITE, height=0.6)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(val + 0.005, bar.get_y() +
                    bar.get_height()/2,
                    f"{val:.4f}",
                    va="center", ha="left",
                    fontsize=9, color="#2D3748")

        ax.axvline(0.5, color="gray", linewidth=1,
                   linestyle="--", alpha=0.6,
                   label="Random baseline (0.5)")
        ax.set_xlim(0, 0.85)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f"{metric} — All 11 Models",
                     fontsize=11, fontweight="bold",
                     color=GREEN, fontfamily= "Times New Roman")
        ax.set_facecolor(LIGHT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE, label="Classical"),
        Patch(facecolor=RED,  label="Deep Learning"),
    ]
    fig.legend(handles=legend_elements,
               loc="lower center", ncol=2,
               fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.05))

    plt.suptitle(
        "Results Summary: All 11 Models\n"
        "JJI VLF | Path-level | 7-day window | "
        "4-hour windows | kernel=6 | event-anchored sequences",
        fontsize=11, fontweight="bold",
        color=NAVY, y=1.02, fontfamily= "Times New Roman"
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summary.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {path}")

def plot_day_analysis(results):
   
    print("\nGenerating day-by-day analysis...")

    from datetime import timedelta

    # Load meta and earthquake days
    meta_test = pd.read_csv(
        os.path.join(OUTPUT_DIR, "meta_test.csv"))
    eq_days   = pd.read_csv(
        os.path.join(OUTPUT_DIR, "earthquake_days.csv"))

    meta_test["date"] = pd.to_datetime(meta_test["date"])

    # Build eq lookup
    eq_lookup = {}
    for _, row in eq_days.iterrows():
        eq_date = pd.to_datetime(row["date"]).date()
        paths   = str(row["matched_paths"])
        if not paths or paths == "nan":
            continue
        for stn in [s.strip() for s in
                    paths.replace("|", ",").split(",")
                    if s.strip()]:
            eq_lookup[(eq_date, stn)] = True

    def get_days_before(row):
        date = row["date"].date()
        stn  = row["station"]
        for d in range(0, 8):
            check = date + timedelta(days=d)
            if (check, stn) in eq_lookup:
                return d
        return -1

    meta_test["days_before"] = meta_test.apply(
        get_days_before, axis=1)

    # Analyse LR and CNN1D
    models_to_analyse = {
        "Logistic Regression": results[0],
        "1D-CNN": next(r for r in results
                       if r["model"] == "10_CNN1D")
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(WHITE)

    for ax, (model_name, r) in zip(
            axes, models_to_analyse.items()):

        if r["model"] in ["10_CNN1D", "11_LSTM"]:
            continue  # skip — need flat meta

        meta_copy = meta_test.copy().reset_index(drop=True)
        meta_copy["y_pred"] = r["y_pred"][:len(meta_copy)]
        meta_copy["y_true"] = r["y_true"][:len(meta_copy)]

        recalls = []
        days    = list(range(0, 8))

        for d in days:
            subset = meta_copy[meta_copy["days_before"] == d]
            if len(subset) == 0:
                recalls.append(0)
                continue
            tp = ((subset["y_true"]==1) &
                  (subset["y_pred"]==1)).sum()
            fn = ((subset["y_true"]==1) &
                  (subset["y_pred"]==0)).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)

        bars = ax.bar([f"Day -{d}" for d in days],
                      recalls,
                      color=[RED if d == 7 else BLUE
                             for d in days],
                      alpha=0.85, edgecolor=WHITE)

        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=9)

        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Recall", fontsize=11)
        ax.set_title(f"{model_name}\nRecall by Days Before Earthquake",
                     fontsize=11, fontweight="bold", color=NAVY)
        ax.set_facecolor(LIGHT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.5, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.5)

    plt.suptitle(
        "Recall by Pre-Seismic Day — LR vs 1D-CNN\n"
        "Day -0 = earthquake day, Day -7 = 7 days before",
        fontsize=11, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "day_analysis.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {path}")
# Main 

def main():
    print("=" * 55)
    print("RESULTS COMPARISON AND VISUALISATION")
    print("=" * 55)

    # 1. Load test data
    X_test, y_test, X_seq, y_seq = load_test_data()

    # 2. Evaluate all models
    results = evaluate_all(X_test, y_test, X_seq, y_seq)

    # 3. Results table
    df, results = save_results_table(results)

    # 4. Bar chart
    plot_bar_chart(results)

    # 5. ROC curves
    plot_roc_curves(results)

    # 6. Confusion matrix — best model
    plot_confusion_matrix(results)

    # 7. Summary figure
    plot_summary(results)
# Add after plot_summary(results):
    plot_day_analysis(results)  
    # Final summary
    best = max(results, key=lambda r: r["f1"])
    print("\n" + "="*55)
    print("COMPLETE")
    print("="*55)
    print(f"  Best model    : {MODEL_NAMES[best['model']]}")
    print(f"  F1-Score      : {best['f1']:.4f}")
    print(f"  AUC-ROC       : {best['auc']:.4f}")
    print(f"  Precision     : {best['precision']:.4f}")
    print(f"  Recall        : {best['recall']:.4f}")
    print(f"\n  Output files:")
    print(f"    results_table.csv")
    print(f"    bar_chart.png")
    print(f"    roc_curves.png")
    print(f"    confusion_matrix.png")
    print(f"    summary.png")
    print(f"\nPipeline complete — ready for thesis writing")


if __name__ == "__main__":
    main()