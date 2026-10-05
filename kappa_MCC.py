import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    cohen_kappa_score, matthews_corrcoef,
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier,
                                   GradientBoostingClassifier)
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.naive_bayes   import GaussianNB

OUTPUT_DIR   = "outputs"
MODELS_DIR   = os.path.join(OUTPUT_DIR, "models")
RANDOM_STATE = 42
N_PERMUTATIONS = 1000
SEQ_LEN      = 42
N_FEATURES   = 4
FLAT_FEATURES = SEQ_LEN * N_FEATURES

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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

#Model definitions
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FLAT_FEATURES, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.network(x).squeeze(1)

class DeepNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FLAT_FEATURES, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.network(x).squeeze(1)

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(N_FEATURES, 32, kernel_size=6, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=6, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1))
    def forward(self, x):
        return self.classifier(
            self.conv_block(x.permute(0, 2, 1))).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, 64, 2,
            batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(1)

#Load data
def load_data():
    X_te_flat = np.load(os.path.join(OUTPUT_DIR, "X_test_flat.npy"))
    X_te_seq  = np.load(os.path.join(OUTPUT_DIR, "X_test_seq.npy"))
    y_te      = np.load(os.path.join(OUTPUT_DIR, "y_test_seq.npy"))
    X_tr_flat = np.load(os.path.join(OUTPUT_DIR, "X_train_flat.npy"))
    X_tr_seq  = np.load(os.path.join(OUTPUT_DIR, "X_train_seq.npy"))
    y_tr      = np.load(os.path.join(OUTPUT_DIR, "y_train_seq.npy"))
    X_v_flat  = np.load(os.path.join(OUTPUT_DIR, "X_val_flat.npy"))
    X_v_seq   = np.load(os.path.join(OUTPUT_DIR, "X_val_seq.npy"))
    y_v       = np.load(os.path.join(OUTPUT_DIR, "y_val_seq.npy"))
    return (X_te_flat, X_te_seq, y_te,
            X_tr_flat, X_tr_seq, y_tr,
            X_v_flat, X_v_seq, y_v)

#Get predictions from saved models
def get_predictions(X_te_flat, X_te_seq, y_te):
    results = {}

    classical = {
        "01_LogisticRegression": None,
        "02_DecisionTree":       None,
        "03_RandomForest":       None,
        "04_GradientBoosting":   None,
        "05_SVM":                None,
        "06_KNN":                None,
        "07_NaiveBayes":         None,
    }
    for name in classical:
        with open(os.path.join(MODELS_DIR,
                  f"{name}.pkl"), "rb") as f:
            model = pickle.load(f)
        yp    = model.predict(X_te_flat)
        yprob = model.predict_proba(X_te_flat)[:, 1]
        results[name] = (yp, yprob)

    dl_flat = [("08_SimpleNN", SimpleNN()),
               ("09_DeepNN",   DeepNN())]
    for name, model in dl_flat:
        model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, f"{name}.pt"),
            map_location="cpu"))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(
                X_te_flat, dtype=torch.float32)).numpy()
        probs = 1 / (1 + np.exp(-logits))
        results[name] = ((probs >= 0.5).astype(int), probs)

    dl_seq = [("10_CNN1D", CNN1D()),
              ("11_LSTM",  LSTMModel())]
    for name, model in dl_seq:
        model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, f"{name}.pt"),
            map_location="cpu"))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(
                X_te_seq, dtype=torch.float32)).numpy()
        probs = 1 / (1 + np.exp(-logits))
        results[name] = ((probs >= 0.5).astype(int), probs)

    return results

#Compute all metrics
def compute_all_metrics(y_true, y_pred, y_prob):
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc"      : round(roc_auc_score(y_true, y_prob), 4),
        "kappa"    : round(cohen_kappa_score(y_true, y_pred), 4),
        "mcc"      : round(matthews_corrcoef(y_true, y_pred), 4),
    }

# Permutation p-value
def compute_pvalue(name, real_f1,
                   X_tr_flat, X_tr_seq, y_tr,
                   X_v_flat, X_v_seq, y_v,
                   X_te_flat, X_te_seq, y_te):
    print(f"    Running {N_PERMUTATIONS} permutations "
          f"for {MODEL_NAMES[name]}...")
    perm_f1s = []
    np.random.seed(RANDOM_STATE)

    for i in range(N_PERMUTATIONS):
        y_rand = np.random.permutation(y_tr)

        if name in ["01_LogisticRegression",
                    "02_DecisionTree",
                    "03_RandomForest",
                    "04_GradientBoosting",
                    "05_SVM","06_KNN",
                    "07_NaiveBayes"]:
            classical_map = {
                "01_LogisticRegression":
                    LogisticRegression(
                        random_state=RANDOM_STATE,
                        max_iter=500),
                "02_DecisionTree":
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                        max_depth=5),
                "03_RandomForest":
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        n_estimators=50, n_jobs=-1),
                "04_GradientBoosting":
                    GradientBoostingClassifier(
                        random_state=RANDOM_STATE,
                        n_estimators=50),
                "05_SVM":
                    SVC(probability=True,
                        random_state=RANDOM_STATE),
                "06_KNN":
                    KNeighborsClassifier(n_jobs=-1),
                "07_NaiveBayes":
                    GaussianNB(),
            }
            m = classical_map[name]
            m.fit(X_tr_flat, y_rand)
            yp = m.predict(X_te_flat)

        else:
            yp = np.random.permutation(y_te)

        perm_f1 = f1_score(
            y_te, yp, zero_division=0)
        perm_f1s.append(perm_f1)

    perm_f1s = np.array(perm_f1s)
    p_value  = (perm_f1s >= real_f1).sum() / N_PERMUTATIONS
    return p_value, perm_f1s

# Main
def main():
    print("="*60)
    print("STATISTICAL VALIDATION")
    print(f"Kappa + MCC + p-value "
          f"({N_PERMUTATIONS} permutations)")
    print("="*60)

    (X_te_flat, X_te_seq, y_te,
     X_tr_flat, X_tr_seq, y_tr,
     X_v_flat, X_v_seq, y_v) = load_data()

    print(f"\n  Test  : {len(y_te)} sequences "
          f"(EQ={y_te.sum()}, "
          f"normal={(y_te==0).sum()})")
    print(f"  Train : {len(y_tr)} sequences")

    # Get predictions
    predictions = get_predictions(
        X_te_flat, X_te_seq, y_te)

    # Compute metrics
    print("\nComputing Kappa and MCC...")
    all_metrics = {}
    for name, (yp, yprob) in predictions.items():
        m = compute_all_metrics(y_te, yp, yprob)
        all_metrics[name] = m
        print(f"  {MODEL_NAMES[name]:<22} "
              f"F1={m['f1']:.4f} "
              f"Kappa={m['kappa']:.4f} "
              f"MCC={m['mcc']:.4f}")

    print(f"\nComputing p-values "
          f"({N_PERMUTATIONS} permutations)...")
    print("  (Classical models retrained each time,")
    print("   Deep learning uses random shuffle "
          "approximation)\n")

    perm_distributions = {}
    start = time.time()

    for name in predictions:
        real_f1 = all_metrics[name]["f1"]
        p_val, perm_f1s = compute_pvalue(
            name, real_f1,
            X_tr_flat, X_tr_seq, y_tr,
            X_v_flat, X_v_seq, y_v,
            X_te_flat, X_te_seq, y_te)
        all_metrics[name]["p_value"] = round(p_val, 4)
        perm_distributions[name] = perm_f1s

        sig = "***" if p_val < 0.001 \
              else "**" if p_val < 0.01 \
              else "*"  if p_val < 0.05 \
              else "ns"
        print(f"  {MODEL_NAMES[name]:<22} "
              f"p={p_val:.4f} {sig}")

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")

    #Results table
    print("\n" + "="*80)
    print("FULL RESULTS TABLE")
    print("="*80)
    print(f"\n  {'Model':<24} {'F1':>7} {'AUC':>7} "
          f"{'Kappa':>7} {'MCC':>7} "
          f"{'p-value':>9} {'Sig':>5}")
    print("  " + "-"*74)

    rows = []
    for name, m in sorted(
            all_metrics.items(),
            key=lambda x: x[1]["f1"],
            reverse=True):
        sig = "***" if m["p_value"] < 0.001 \
              else "**"  if m["p_value"] < 0.01 \
              else "*"   if m["p_value"] < 0.05 \
              else "ns"
        print(f"  {MODEL_NAMES[name]:<24} "
              f"{m['f1']:>7.4f} "
              f"{m['auc']:>7.4f} "
              f"{m['kappa']:>7.4f} "
              f"{m['mcc']:>7.4f} "
              f"{m['p_value']:>9.4f} "
              f"{sig:>5}")
        rows.append({
            "Model"    : MODEL_NAMES[name],
            "F1"       : m["f1"],
            "AUC"      : m["auc"],
            "Kappa"    : m["kappa"],
            "MCC"      : m["mcc"],
            "p_value"  : m["p_value"],
            "Sig"      : sig,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(
        OUTPUT_DIR, "statistical_validation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    #Visualisation
    print("\nGenerating visualisation...")

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("white")

    NAVY  = "#1B3A5C"
    LIGHT = "#F7FAFD"
    names_short = [MODEL_NAMES[n].replace(
        " ", "\n") for n in all_metrics]

    order = sorted(all_metrics.keys(),
                   key=lambda n: all_metrics[n]["f1"],
                   reverse=True)
    names_ord = [MODEL_NAMES[n] for n in order]

    ax1 = fig.add_subplot(2, 2, 1)
    x   = np.arange(len(order))
    w   = 0.25

    f1s    = [all_metrics[n]["f1"]    for n in order]
    kappas = [all_metrics[n]["kappa"] for n in order]
    mccs   = [all_metrics[n]["mcc"]   for n in order]

    ax1.bar(x - w, f1s,    w, label="F1-Score",
            color="#2E6DA4", alpha=0.85,
            edgecolor="white")
    ax1.bar(x,     kappas, w, label="Kappa",
            color="#2E7D52", alpha=0.85,
            edgecolor="white")
    ax1.bar(x + w, mccs,   w, label="MCC",
            color="#C0392B", alpha=0.85,
            edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [n.replace(" ", "\n") for n in names_ord],
        fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title(
        "F1, Cohen's Kappa and MCC\nAll 11 Models",
        fontsize=11, fontweight="bold", color=NAVY, fontname = "Times New Roman")
    ax1.legend(fontsize=9)
    ax1.axhline(0.5, color="gray",
                linewidth=0.8, linestyle="--",
                alpha=0.5)
    ax1.set_facecolor(LIGHT)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)
    ax2 = fig.add_subplot(2, 2, 2)
# p - value
    p_vals  = [all_metrics[n]["p_value"]
               for n in order]
    colors2 = ["#C0392B" if p < 0.05
               else "#888" for p in p_vals]

    bars = ax2.bar(
        [n.replace(" ", "\n") for n in names_ord],
        p_vals, color=colors2,
        alpha=0.85, edgecolor="white",
        width=0.6)

    for bar, p in zip(bars, p_vals):
        sig = "***" if p < 0.001 \
              else "**"  if p < 0.01 \
              else "*"   if p < 0.05 \
              else "ns"
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{sig}\np={p:.3f}",
            ha="center", fontsize=8,
            color="#2D3748")

    ax2.axhline(0.05, color="#C0392B",
                linewidth=1.2, linestyle="--",
                alpha=0.8,
                label="p=0.05 significance threshold")
    ax2.set_ylabel("p-value", fontsize=10)
    ax2.set_title(
        f"Permutation Test p-values\n"
        f"({N_PERMUTATIONS} permutations)",
        fontsize=11, fontweight="bold", color=NAVY,fontname = "Times New Roman"),
    ax2.legend(fontsize=8)
    ax2.set_facecolor(LIGHT)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=8)

    # Permutation distribution for GB
    ax3 = fig.add_subplot(2, 2, 3)

    gb_perm  = perm_distributions[
        "04_GradientBoosting"]
    gb_real  = all_metrics[
        "04_GradientBoosting"]["f1"]
    gb_pval  = all_metrics[
        "04_GradientBoosting"]["p_value"]

    ax3.hist(gb_perm, bins=40,
             color="#2E7D52", alpha=0.7,
             edgecolor="white",
             label=f"Random label F1\n"
                   f"(n={N_PERMUTATIONS})")
    ax3.axvline(gb_real, color="#C0392B",
                linewidth=2.5,
                label=f"Real F1={gb_real:.4f}")
    ax3.axvline(np.mean(gb_perm),
                color="#2E6DA4",
                linewidth=1.5, linestyle="--",
                label=f"Mean random="
                      f"{np.mean(gb_perm):.4f}")

    ax3.set_xlabel("F1-Score", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title(
        "Gradient Boosting — Permutation Distribution\n"
        f"p={gb_pval:.4f} — real F1 vs random",
        fontsize=11, fontweight="bold", color=NAVY, fontname = "Times New Roman")
    ax3.legend(fontsize=9)
    ax3.set_facecolor(LIGHT)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Permutation distribution for CNN1D
    ax4 = fig.add_subplot(2, 2, 4)

    cnn_perm = perm_distributions["10_CNN1D"]
    cnn_real = all_metrics["10_CNN1D"]["f1"]
    cnn_pval = all_metrics["10_CNN1D"]["p_value"]

    ax4.hist(cnn_perm, bins=40,
             color="#2E6DA4", alpha=0.7,
             edgecolor="white",
             label=f"Random label F1\n"
                   f"(n={N_PERMUTATIONS})")
    ax4.axvline(cnn_real, color="#C0392B",
                linewidth=2.5,
                label=f"Real F1={cnn_real:.4f}")
    ax4.axvline(np.mean(cnn_perm),
                color="#2E7D52",
                linewidth=1.5, linestyle="--",
                label=f"Mean random="
                      f"{np.mean(cnn_perm):.4f}")

    ax4.set_xlabel("F1-Score", fontsize=10)
    ax4.set_ylabel("Count", fontsize=10)
    ax4.set_title(
        "1D-CNN — Permutation Distribution\n"
        f"p={cnn_pval:.4f} — real F1 vs random",
        fontsize=11, fontweight="bold", color=NAVY, fontname= "Times New Roman")
    ax4.legend(fontsize=9)
    ax4.set_facecolor(LIGHT)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    plt.suptitle(
        "Statistical Validation — All 11 Models\n"
        "Cohen's Kappa · MCC · "
        f"Permutation p-value ({N_PERMUTATIONS} runs)",
        fontsize=13, fontweight="bold",
        color=NAVY, y=1.01)

    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR, "statistical_validation.png")
    plt.savefig(path, dpi=150,
                bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

    print("\n" + "="*60)
    print("SIGNIFICANCE LEGEND")
    print("="*60)
    print("  *** p < 0.001  highly significant")
    print("  **  p < 0.010  very significant")
    print("  *   p < 0.050  significant")
    print("  ns  p >= 0.050 not significant")

    print("\n Statistical validation complete")

if __name__ == "__main__":
    main()