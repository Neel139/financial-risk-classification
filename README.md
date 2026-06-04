# Financial Risk Classification — Company Bankruptcy Prediction

A multi-stage machine learning pipeline to predict company bankruptcy from 95 financial indicators. The core challenge: the dataset is severely imbalanced (~3% bankrupt), making standard classifiers fail entirely. This project addresses that through unsupervised clustering, SMOTE oversampling, and per-cluster stacking ensembles.

---

## The Problem

A baseline logistic regression on the raw data achieves **0% bankruptcy detection accuracy on the test set** — it simply predicts everyone as non-bankrupt. The goal was to build a system that actually identifies the companies that will go bankrupt.

---

## Pipeline Overview

```
95 Raw Features
      │
      ▼
Feature Engineering & Selection
(Correlation pruning → SelectKBest → Gaussian normalization → 13 features)
      │
      ▼
KMeans Clustering (k=5)
(Group companies by financial profile)
      │
      ├── Cluster 0 → Stacking Ensemble (Katherine)
      ├── Cluster 1 → Stacking Ensemble (Neel)      ← highest bankruptcy rate
      ├── Cluster 2 → Stacking Ensemble (Jorge)
      ├── Cluster 3 → Constant Model (ĥ = 0)        ← no bankruptcies
      └── Cluster 4 → Stacking Ensemble (Mahad)
      │
      ▼
Per-Cluster Stacking Model
(Base models → Meta model → Final prediction)
      │
      ▼
Bankruptcy Prediction (1012 test companies)
```

---

## Cluster Summary

| Cluster | Companies | Bankrupted | Bankruptcy Rate |
|---------|-----------|------------|-----------------|
| 0 | 1,122 | 156 | 13.9% |
| 1 | 842 | 11 | 1.3% |
| 2 | 858 | 17 | 2.0% |
| 3 | 1,143 | 0 | 0.0% (constant model) |
| 4 | 1,842 | 14 | 0.8% |

Cluster 0 has by far the highest bankruptcy concentration — companies in this group share characteristics of high debt ratios and low cash flow, making them the most financially distressed group.

---

## My Contribution — Subgroup 1

**Feature selection:** Used a dual-signal approach combining Random Forest feature importance and Mutual Information scoring to select the 4 most predictive features:
- `Operating Profit Per Person`
- `ROA(C) before interest and depreciation`
- `Net Value Per Share (B)`
- `Working Capital to Total Assets`

**Model:** Stacking classifier with 4 non-parametric base models + Logistic Regression meta-model

| Base Model | Accuracy (TT/TF) |
|------------|-----------------|
| Bagging (300 estimators) | 1.00 [11(0)] |
| Extra Trees | 1.00 [11(0)] |
| Gradient Boosting | 1.00 [11(0)] |
| KNN (k=5, distance-weighted) | 1.00 [11(0)] |
| **Stacking Meta-Model** | **1.00 [11(0)]** |

**Key techniques:**
- SMOTE oversampling to handle class imbalance before training
- StratifiedKFold (5-fold) cross-validation in stacking
- Custom probability threshold (0.30) tuned for recall on the minority class

---

## Final Predictions

The generalization pipeline predicted **20 out of 1,012 test companies** (~2%) as likely to file for bankruptcy — consistent with the ~3% base rate in training data.

---

## Tech Stack

- **ML:** scikit-learn (StackingClassifier, RandomForest, ExtraTrees, GradientBoosting, KNN, BaggingClassifier)
- **Imbalanced learning:** imbalanced-learn (SMOTE)
- **Feature engineering:** QuantileTransformer, SelectKBest, PCA
- **Visualization:** Matplotlib, Seaborn
- **Data:** pandas, NumPy

---

## Project Structure

```
financial-risk-classification/
├── GroupNumber4_TrainingData.ipynb     # Feature engineering, clustering (Section 3.1 & 3.2)
├── Neel_Subgroup1.ipynb                # Stacking model for Cluster 1 (my work)
├── JorgeP_SubGroup2.ipynb              # Stacking model for Cluster 2
├── KatherineShagalov_Subgroup0.ipynb   # Stacking model for Cluster 0
├── MahadR_SubGroup4.ipynb              # Stacking model for Cluster 4
├── Group4_Generalization.ipynb         # Test set predictions
├── GroupNumber_Generalization.csv      # Final submission (1012 predictions)
└── GroupNumber4_CS559WS_Results.docx   # Full results report
```

---

## Key Takeaways

- Standard classifiers completely fail on severely imbalanced financial data — domain-aware restructuring (clustering by financial profile) is essential
- Combining unsupervised and supervised learning in a pipeline significantly improves minority class detection
- Feature selection with dual signals (RF importance + Mutual Information) is more robust than either alone
