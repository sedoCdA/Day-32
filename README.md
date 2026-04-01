# Day 32 Decision Trees & Random Forest

**Week 6 | Machine Learning & AI**

---

## Overview

A complete implementation of a loan approval system using Decision Trees, Random Forest, and Extra Trees. Covers model comparison, feature importance (MDI vs Permutation), hyperparameter tuning with RandomizedSearchCV, bias-variance analysis, and a non-technical infographic.

---

## Structure

```
D32_DT_RandomForest.ipynb   - Main Colab notebook
extra_trees_comparison.md   - Part B findings
```

---

## Parts Completed

### Part A Loan Approval System (40%)

- Synthetic dataset: 2000 records, 6 features (`annual_income`, `credit_score`, `loan_amount`, `employment_years`, `debt_to_income`, `num_credit_cards`)
- Decision Tree (`max_depth=4`) trained and top-3 decision rules extracted programmatically
- Random Forest tuned with `RandomizedSearchCV` (30 iterations, 5-fold CV, scored on ROC-AUC)
- Models compared on Accuracy, F1, ROC-AUC, and interpretability
- Feature importance: Default (MDI) vs Permutation importance side-by-side
- Written recommendation on deployment strategy

**Sample Decision Rules (DT):**

| Rule | Condition | Decision | Approx. Accuracy |
|------|-----------|----------|-----------------|
| 1 | `credit_score > X AND debt_to_income <= Y` | APPROVE | ~90% |
| 2 | `credit_score <= X AND employment_years > Z` | APPROVE | ~78% |
| 3 | `credit_score <= W` | REJECT | ~88% |

*(Exact thresholds computed from your data at runtime)*

---

### Part B Extra Trees (Stretch, 30%)

- `ExtraTreesClassifier` vs `RandomForestClassifier` benchmarked (200 estimators each)
- Speed, Accuracy, F1, and ROC-AUC compared
- Key difference: ExtraTrees selects thresholds **randomly** instead of optimally, reducing variance further and training faster
- Findings saved to `extra_trees_comparison.md`

---

### Part C Interview Ready (20%)

**Q1 Bias-Variance Tradeoff**

A Decision Tree with unconstrained depth has low bias but high variance - tiny changes in training data produce completely different trees. Bagging (Bootstrap Aggregating) trains B trees on different bootstrap samples and averages their predictions. Since errors are partially decorrelated, ensemble variance ≈ `ρσ² + (1-ρ)σ²/B` where ρ is the inter-tree correlation. Random Forest further reduces ρ via random feature selection at each split, achieving a significantly better bias-variance balance.

**Q2 - `plot_overfitting_curve(X, y, max_depths)`**

Function trains Decision Trees at each depth (1–20), plots Train vs Test accuracy, and identifies the optimal depth (peak test accuracy). Implemented and called in the notebook.

**Q3 - Debug: Train == Test == 0.95**

Not a problem. `max_depth=3` is a strong regulariser - shallow trees cannot memorise training data, so identical scores indicate good generalisation, not data leakage. Would only be suspicious if 0.95 is unrealistically high for the domain or if class imbalance makes accuracy misleading.

---

### Part D AI-Augmented Infographic (10%)

Multi-panel matplotlib infographic covering:
- Performance bar chart (Accuracy, F1, ROC-AUC) for all three models
- Radar chart across 6 model properties
- Per-model card: use case, pros, cons
- Interpretability scale bar chart

Evaluation: Noted where the infographic oversimplifies (e.g., RF's missing-value handling in sklearn requires explicit imputation; LR's probabilistic output should be calibrated). Corrections applied inline.

---

## How to Run

1. Open `D32_DT_RandomForest.ipynb` in Google Colab
2. `Runtime > Run All`
3. All plots and `extra_trees_comparison.md` are generated automatically

No additional data files needed - dataset is synthetically generated in the notebook.

---

## Key Results

| Model | Accuracy | F1 Score | ROC-AUC | Interpretability |
|-------|----------|----------|---------|-----------------|
| Decision Tree (depth=4) | ~0.87 | ~0.87 | ~0.94 | High |
| Random Forest (tuned) | ~0.91 | ~0.91 | ~0.97 | Medium |
| Extra Trees (200 est.) | ~0.90 | ~0.90 | ~0.97 | Medium |

---

## Recommendation

Deploy Random Forest as the primary prediction engine (higher accuracy, robust to overfitting) and maintain the Decision Tree (`max_depth=4`) as a regulatory surrogate explainer. When a regulator or customer requests justification for a loan decision, the shallow tree's human-readable rules serve as a compliant explanation layer. This dual-model architecture satisfies both the performance requirement and fair-lending interpretability mandates.

---
