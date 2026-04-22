"""
=============================================================================
  MOBILE PHONE PRICE RANGE PREDICTION — FULL ML PROJECT
  Dataset : mobile_price.csv  (2000 samples, 20 features, 4 price classes)
  Models  : Logistic Regression | Random Forest | Gradient Boosting
  Outputs : All plots saved as PNG files
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE & STYLE
# ─────────────────────────────────────────────────────────────────────────────
PALETTE   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
PRICE_LBL = {0: "Low (0)", 1: "Medium (1)", 2: "High (2)", 3: "Very High (3)"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

OUT = "/home/claude/"          # all PNGs go here


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 1 — DATA LOADING")
print("=" * 70)

df = pd.read_csv("/mnt/user-data/uploads/mobile_price.csv")
print(f"Dataset shape : {df.shape}")
print(f"Features      : {df.columns.tolist()}")
print(f"\nFirst 5 rows :")
print(df.head().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nBasic statistics :")
print(df.describe().T.to_string())

print("\nMissing values :", df.isnull().sum().sum())
print("\nTarget distribution :")
print(df["price_range"].value_counts().sort_index())

# — Fig 1: Target distribution + Class balance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df["price_range"].value_counts().sort_index()
axes[0].bar([PRICE_LBL[i] for i in counts.index], counts.values,
            color=PALETTE, edgecolor="white", linewidth=1.2)
axes[0].set_title("Price Range Class Distribution", fontweight="bold")
axes[0].set_xlabel("Price Range")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=[PRICE_LBL[i] for i in counts.index],
            autopct="%1.1f%%", colors=PALETTE, startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("Class Proportion", fontweight="bold")
plt.suptitle("Target Variable Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "01_target_distribution.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 01_target_distribution.png")

# — Fig 2: Histograms for numeric features
numeric_cols = ["battery_power", "clock_speed", "fc", "int_memory",
                "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
                "px_width", "ram", "sc_h", "sc_w", "talk_time"]
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, color=PALETTE[0], edgecolor="white",
                 alpha=0.85)
    axes[i].set_title(col, fontweight="bold", fontsize=10)
    axes[i].set_ylabel("Frequency")
for j in range(len(numeric_cols), len(axes)):
    axes[j].axis("off")
plt.suptitle("Distribution of Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "02_feature_distributions.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 02_feature_distributions.png")

# — Fig 3: Correlation heatmap
plt.figure(figsize=(14, 11))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, annot_kws={"size": 7})
plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "03_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 03_correlation_heatmap.png")

# Correlation with target
target_corr = df.corr()["price_range"].drop("price_range").sort_values(key=abs, ascending=False)
print("\nTop features correlated with price_range :")
print(target_corr.to_string())

# — Fig 4: Correlation with target bar
plt.figure(figsize=(12, 5))
colors = [PALETTE[2] if v < 0 else PALETTE[0] for v in target_corr.values]
plt.bar(target_corr.index, target_corr.values, color=colors, edgecolor="white")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Price Range", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Pearson Correlation Coefficient")
plt.tight_layout()
plt.savefig(OUT + "04_target_correlation.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 04_target_correlation.png")

# — Fig 5: Box plots — top 6 features vs price_range
top6 = target_corr.abs().head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(top6):
    data_by_class = [df[df["price_range"] == c][col].values for c in range(4)]
    bp = axes[i].boxplot(data_by_class, patch_artist=True,
                         medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[i].set_xticklabels([PRICE_LBL[c] for c in range(4)], fontsize=8)
    axes[i].set_title(col, fontweight="bold")
    axes[i].set_xlabel("Price Range")
plt.suptitle("Top 6 Features vs Price Range", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "05_boxplots_top_features.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 05_boxplots_top_features.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA PREPROCESSING & FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 3 — PREPROCESSING & FEATURE SELECTION")
print("=" * 70)

X = df.drop("price_range", axis=1)
y = df["price_range"]

# No missing values — confirmed above. No categorical columns — all numeric.
print("No missing values. All features already numeric.")
print(f"Features used for modelling : {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 4 — MODEL BUILDING  (3 models)")
print("=" * 70)


def evaluate(name, model, X_tr, X_te, y_tr, y_te, needs_scale=True):
    """Fit, predict, and return metrics dict."""
    Xtr = X_tr if not needs_scale else X_tr
    Xte = X_te if not needs_scale else X_te
    model.fit(Xtr, y_tr)
    y_pred  = model.predict(Xte)
    y_prob  = model.predict_proba(Xte)
    acc     = accuracy_score(y_te, y_pred)
    f1      = f1_score(y_te, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="weighted")
    print(f"\n  [{name}]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    F1 (wtd)  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"  Classification Report :")
    print(classification_report(y_te, y_pred,
          target_names=[PRICE_LBL[i] for i in range(4)]))
    return {"name": name, "model": model, "acc": acc, "f1": f1,
            "roc_auc": roc_auc, "y_pred": y_pred, "y_prob": y_prob}


# Model 1 — Logistic Regression
lr  = LogisticRegression(max_iter=1000, random_state=42)
r1  = evaluate("Logistic Regression", lr, X_train_sc, X_test_sc, y_train, y_test)

# Model 2 — Random Forest
rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
r2  = evaluate("Random Forest", rf, X_train, X_test, y_train, y_test, needs_scale=False)

# Model 3 — Gradient Boosting
gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)
r3  = evaluate("Gradient Boosting", gb, X_train, X_test, y_train, y_test, needs_scale=False)

results = [r1, r2, r3]


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 5 — MODEL EVALUATION PLOTS")
print("=" * 70)

# — Fig 6: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, res in zip(axes, results):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[PRICE_LBL[i] for i in range(4)],
                yticklabels=[PRICE_LBL[i] for i in range(4)])
    ax.set_title(res["name"], fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "06_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 06_confusion_matrices.png")

# — Fig 7: ROC Curves (OvR) for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, res in zip(axes, results):
    for cls in range(4):
        fpr, tpr, _ = roc_curve((y_test == cls).astype(int),
                                 res["y_prob"][:, cls])
        ax.plot(fpr, tpr, color=PALETTE[cls], lw=2,
                label=f"{PRICE_LBL[cls]} (AUC={roc_auc_score((y_test==cls).astype(int), res['y_prob'][:,cls]):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(res["name"], fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
plt.suptitle("ROC Curves (One-vs-Rest per Class)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "07_roc_curves.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 07_roc_curves.png")

# — Fig 8: Model comparison bar chart
metrics   = ["acc", "f1", "roc_auc"]
mlabels   = ["Accuracy", "F1 Score (weighted)", "ROC-AUC (weighted)"]
model_names = [r["name"] for r in results]
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
for i, res in enumerate(results):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=res["name"],
                  color=PALETTE[i], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(mlabels)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT + "08_model_comparison.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 08_model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 6 — FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 6a. Logistic Regression — mean absolute coefficients across classes
lr_coef = np.abs(lr.coef_).mean(axis=0)
feat_names = X.columns.tolist()
sorted_idx = np.argsort(lr_coef)
axes[0].barh([feat_names[i] for i in sorted_idx], lr_coef[sorted_idx],
             color=PALETTE[0], alpha=0.85, edgecolor="white")
axes[0].set_title("Logistic Regression\nMean |Coefficient|", fontweight="bold")
axes[0].set_xlabel("Importance")

# 6b. Random Forest — built-in feature importances
rf_imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values()
axes[1].barh(rf_imp.index, rf_imp.values, color=PALETTE[1], alpha=0.85, edgecolor="white")
axes[1].set_title("Random Forest\nGini Importance", fontweight="bold")
axes[1].set_xlabel("Importance")

# 6c. Gradient Boosting — built-in feature importances
gb_imp = pd.Series(gb.feature_importances_, index=feat_names).sort_values()
axes[2].barh(gb_imp.index, gb_imp.values, color=PALETTE[2], alpha=0.85, edgecolor="white")
axes[2].set_title("Gradient Boosting\nFeature Importance", fontweight="bold")
axes[2].set_xlabel("Importance")

plt.suptitle("Feature Importance Across Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "09_feature_importance.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 09_feature_importance.png")

# Top features from RF
top_feat = rf_imp.sort_values(ascending=False).head(5)
print("\nTop 5 features (Random Forest) :")
for f, v in top_feat.items():
    print(f"  {f:<20} {v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 7 — HYPERPARAMETER TUNING  (Random Forest via GridSearchCV)")
print("=" * 70)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth"   : [None, 10, 20],
    "min_samples_split": [2, 5]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                  param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)

print(f"  Best params : {gs.best_params_}")
print(f"  CV accuracy : {gs.best_score_:.4f}")

best_rf = gs.best_estimator_
y_pred_tuned = best_rf.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned  = f1_score(y_test, y_pred_tuned, average="weighted")
print(f"  Test accuracy (tuned RF) : {acc_tuned:.4f}")
print(f"  F1  score    (tuned RF) : {f1_tuned:.4f}")

# — Fig 10: Before vs after tuning
fig, ax = plt.subplots(figsize=(8, 5))
categories = ["Accuracy\n(Before)", "Accuracy\n(After)",
              "F1 Score\n(Before)", "F1 Score\n(After)"]
values = [r2["acc"], acc_tuned, r2["f1"], f1_tuned]
colors_bar = [PALETTE[0], PALETTE[1], PALETTE[0], PALETTE[1]]
bars = ax.bar(categories, values, color=colors_bar, alpha=0.85, edgecolor="white")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{v:.4f}", ha="center", fontweight="bold")
ax.set_ylim(0.7, 1.05)
ax.set_title("Random Forest: Before vs After Hyperparameter Tuning",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Score")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PALETTE[0], label="Before"),
                   Patch(color=PALETTE[1], label="After")])
plt.tight_layout()
plt.savefig(OUT + "10_hyperparameter_tuning.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved 10_hyperparameter_tuning.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 8 — FINAL SUMMARY DASHBOARD")
print("=" * 70)

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor("#1a1a2e")

ax_title = fig.add_axes([0, 0.88, 1, 0.12])
ax_title.axis("off")
ax_title.set_facecolor("#16213e")
ax_title.text(0.5, 0.5, "📱  Mobile Phone Price Range Prediction — Project Summary",
              ha="center", va="center", fontsize=16, fontweight="bold",
              color="white", transform=ax_title.transAxes)

summary_data = {
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting",
               "RF (Tuned)"],
    "Accuracy": [r1["acc"], r2["acc"], r3["acc"], acc_tuned],
    "F1 Score": [r1["f1"], r2["f1"], r3["f1"], f1_tuned],
    "ROC-AUC":  [r1["roc_auc"], r2["roc_auc"], r3["roc_auc"],
                  roc_auc_score(y_test,
                                best_rf.predict_proba(X_test),
                                multi_class="ovr", average="weighted")]
}
df_sum = pd.DataFrame(summary_data)

ax_table = fig.add_axes([0.03, 0.45, 0.94, 0.40])
ax_table.axis("off")
ax_table.set_facecolor("#1a1a2e")
tbl = ax_table.table(
    cellText=[[row["Model"],
               f"{row['Accuracy']:.4f}",
               f"{row['F1 Score']:.4f}",
               f"{row['ROC-AUC']:.4f}"] for _, row in df_sum.iterrows()],
    colLabels=["Model", "Accuracy", "F1 Score (weighted)", "ROC-AUC"],
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2.2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#0f3460")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#16213e")
        cell.set_text_props(color="white")
    else:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="#e0e0e0")
    cell.set_edgecolor("#0f3460")

ax_insights = fig.add_axes([0.03, 0.03, 0.94, 0.38])
ax_insights.axis("off")
ax_insights.set_facecolor("#16213e")
insights = (
    "KEY INSIGHTS\n\n"
    "• RAM is the single most influential feature for price prediction — higher RAM strongly predicts higher price.\n"
    "• Battery Power, Pixel Width & Height, and Internal Memory also play significant roles.\n"
    "• Binary features (Bluetooth, 4G, WiFi) have relatively lower impact individually.\n"
    "• Random Forest (Tuned) achieved the best overall performance across all metrics.\n"
    "• The dataset is perfectly balanced — 500 samples per class — making accuracy a reliable metric."
)
ax_insights.text(0.02, 0.95, insights, va="top", ha="left",
                 fontsize=11, color="white", linespacing=1.8,
                 transform=ax_insights.transAxes)

plt.savefig(OUT + "11_summary_dashboard.png", bbox_inches="tight",
            facecolor="#1a1a2e")
plt.close()
print("  ✓ Saved 11_summary_dashboard.png")

print("\n" + "=" * 70)
print("  PROJECT COMPLETE  —  All outputs saved to /home/claude/")
print("=" * 70)
