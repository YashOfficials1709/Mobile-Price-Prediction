# 📱 Mobile Phone Price Range Prediction

A complete machine learning project that predicts the price range of mobile phones based on their technical specifications. Built using Python and scikit-learn.

---

## 📋 Project Overview

This project was developed for a mobile phone retail organization to enhance its pricing strategy. By analyzing key hardware features, the model predicts which of four price categories a phone falls into — **Low**, **Medium**, **High**, or **Very High**.

The full pipeline covers data exploration, preprocessing, feature importance analysis, multi-model training, evaluation, and hyperparameter tuning.

---

## 📁 Project Structure

```
mobile-price-prediction/
│
├── mobile_price_prediction.py   # Main project script (end-to-end pipeline)
├── mobile_price.csv             # Dataset (2000 samples, 20 features)
├── README.md                    # This file
│
└── outputs/
    ├── 01_target_distribution.png
    ├── 02_feature_distributions.png
    ├── 03_correlation_heatmap.png
    ├── 04_target_correlation.png
    ├── 05_boxplots_top_features.png
    ├── 06_confusion_matrices.png
    ├── 07_roc_curves.png
    ├── 08_model_comparison.png
    ├── 09_feature_importance.png
    ├── 10_hyperparameter_tuning.png
    └── 11_summary_dashboard.png
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `mobile_price.csv` |
| Samples | 2,000 |
| Features | 20 |
| Target | `price_range` (0 = Low, 1 = Medium, 2 = High, 3 = Very High) |
| Class balance | Perfectly balanced — 500 samples per class |
| Missing values | None |

### Feature List

| Feature | Description |
|---|---|
| `battery_power` | Battery capacity (mAh) |
| `blue` | Bluetooth support (0/1) |
| `clock_speed` | Processor clock speed (GHz) |
| `dual_sim` | Dual SIM support (0/1) |
| `fc` | Front camera megapixels |
| `four_g` | 4G support (0/1) |
| `int_memory` | Internal memory (GB) |
| `m_dep` | Mobile depth (cm) |
| `mobile_wt` | Weight (grams) |
| `n_cores` | Number of processor cores |
| `pc` | Primary (rear) camera megapixels |
| `px_height` | Screen pixel height |
| `px_width` | Screen pixel width |
| `ram` | RAM (MB) |
| `sc_h` | Screen height (cm) |
| `sc_w` | Screen width (cm) |
| `talk_time` | Battery talk time (hours) |
| `three_g` | 3G support (0/1) |
| `touch_screen` | Touchscreen (0/1) |
| `wifi` | WiFi support (0/1) |

---

## 🔧 Requirements

```
Python >= 3.8
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install dependencies:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## 🚀 How to Run

```bash
python mobile_price_prediction.py
```

All output plots are saved to the working directory automatically.

---

## 🤖 Models Built

Three classification models were trained and compared:

| Model | Accuracy | F1 Score (weighted) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | **96.50%** | **0.9650** | **0.9987** |
| Gradient Boosting | 91.25% | 0.9123 | 0.9896 |
| Random Forest | 88.00% | 0.8797 | 0.9769 |
| Random Forest (Tuned) | 87.75% | 0.8774 | 0.9820 |

**Winner: Logistic Regression** — achieves the highest accuracy due to RAM's strong linear relationship with price range (Pearson r = 0.917).

---

## 🔍 Key Findings

### Most Important Features

1. **RAM** — By far the strongest predictor (Gini importance: 0.48, Pearson r: 0.917)
2. **Battery Power** — Second most influential (r = 0.20)
3. **Pixel Width** — Screen resolution matters (r = 0.17)
4. **Pixel Height** — Consistent with px_width (r = 0.15)
5. **Internal Memory** — Higher storage correlates with higher price

### Key Observations

- Binary features (Bluetooth, WiFi, 4G, Dual SIM) have individually low correlation with price.
- Mobile weight is slightly negatively correlated — lighter phones tend to be more expensive.
- Clock speed has almost no predictive power on its own.
- The "Medium" and "High" price classes are hardest to distinguish across all models.

---

## 📈 Pipeline Steps

1. **Data Loading & Exploration** — Shape, types, distributions, missing value check
2. **EDA** — Histograms, correlation heatmap, box plots per price class
3. **Preprocessing** — StandardScaler for Logistic Regression; 80/20 stratified split
4. **Feature Selection** — Pearson correlation + Gini importance from tree models
5. **Model Building** — Logistic Regression, Random Forest, Gradient Boosting
6. **Evaluation** — Accuracy, F1, ROC-AUC, confusion matrices, ROC curves
7. **Feature Importance Analysis** — Compared across all three models
8. **Hyperparameter Tuning** — GridSearchCV on Random Forest (5-fold stratified CV)

---

## 💡 Recommendations

Based on the findings, the following is recommended to the organization:

- **Prioritize RAM** as the primary pricing signal — it alone explains most of the price variance.
- **Bundle battery capacity** into pricing tiers as the second strongest differentiator.
- **Screen resolution** (pixel width × height) should factor into mid-to-high tier pricing.
- Binary features like 4G/WiFi/Bluetooth are now commodity features and have diminishing impact on perceived price.
- Use **Logistic Regression** as the production model — it is fast, interpretable, and achieved the highest accuracy (96.5%).

---

## 📚 References

- [Feature Extraction Techniques — Towards Data Science](https://towardsdatascience.com)
- [scikit-learn Documentation](https://scikit-learn.org)
- Project brief: `ML___AI_Project.pdf`
- Data dictionary: `Data_Dictionary_NaiveBayes.docx`
- Problem statement: `Problem_Statement_NaiveBayes.docx`

---

## 👤 Author

Developed as part of an ML/AI course project.  
Dataset: Mobile Price Classification (Kaggle)
