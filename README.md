# Day 32 Decision Trees & Random Forest

**Week 6 - Day 32 - AM/PM**

---

## Overview

A complete implementation of a loan approval system using Decision Trees, Random Forest, and Extra Trees. Covers model comparison, feature importance (MDI vs Permutation), hyperparameter tuning with RandomizedSearchCV, bias-variance analysis, and a non-technical infographic.

---

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

### Part C 
**Q1 Bias-Variance Tradeoff**

A Decision Tree with unconstrained depth has low bias but high variance - tiny changes in training data produce completely different trees. Bagging (Bootstrap Aggregating) trains B trees on different bootstrap samples and averages their predictions. Since errors are partially decorrelated, ensemble variance ≈ `ρσ² + (1-ρ)σ²/B` where ρ is the inter-tree correlation. Random Forest further reduces ρ via random feature selection at each split, achieving a significantly better bias-variance balance.

**Q2 - `plot_overfitting_curve(X, y, max_depths)`**

Function trains Decision Trees at each depth (1–20), plots Train vs Test accuracy, and identifies the optimal depth (peak test accuracy). Implemented and called in the notebook.

**Q3 - Debug: Train == Test == 0.95**

Not a problem. `max_depth=3` is a strong regulariser - shallow trees cannot memorise training data, so identical scores indicate good generalisation, not data leakage. Would only be suspicious if 0.95 is unrealistically high for the domain or if class imbalance makes accuracy misleading.

---

### Part D 

Multi-panel matplotlib infographic covering:
- Performance bar chart (Accuracy, F1, ROC-AUC) for all three models
- Radar chart across 6 model properties
- Per-model card: use case, pros, cons
- Interpretability scale bar chart

Evaluation: Noted where the infographic oversimplifies (e.g., RF's missing-value handling in sklearn requires explicit imputation; LR's probabilistic output should be calibrated). Corrections applied inline.

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
# Day 32 PM Session: Decision Trees & Random Forest Applied
This documentation provides the conceptual analysis and technical justifications for the Day 32 PM Case Study on insurance fraud detection.

---

## Part A: Deployment Strategy & Business Logic
**Scenario:** Predicting insurance fraud where a missed fraudster (False Negative) costs 10x more than a false alarm (False Positive).

### Recommendation
**Automated Scoring:** The **Random Forest** (RF) model is the superior choice for the primary engine By tuning the model specifically to maximize **Recall**, we ensure the bank captures the highest percentage of fraudulent claims, directly addressing the high cost of False Negatives.
**Regulatory Compliance:** To meet requirements for explainability, we utilize the **Decision Tree** (DT) as a "Proxy Model" While the RF makes the high-accuracy decision, the DT provides a visual, rule-based map (e.g., "If claim > $5000 and Emergency = Yes") that adjusters can use to justify investigations to regulators.

---

## Part B: Ensemble Methods (Bagging vs. Boosting)
**Conceptual Comparison:**
**Bagging (Random Forest):** Operates by building many trees independently in parallel. Each tree gets a random subset of data (bootstrap sampling). The final prediction is an average, which primarily serves to reduce **variance** and overfitting.
**Boosting:** Operates sequentially. Each new tree is trained specifically to predict the errors (residuals) made by the previous trees. This process focuses on reducing **bias**, turning a collection of weak models into one powerful learner.

---

## Part C
### Q1: Efficiency and Scaling
**Question:** If 1000 trees yield the same accuracy as 100 trees, which do you deploy?
**Answer:** Deploy the **100-tree model**.
**Computational Overhead:** 1000 trees require 10x the memory and processing power to train.
* **Latency:** In production, every additional tree adds time to the prediction "forward pass."For real-time insurance claims, lower latency is critical.
* **Complexity:** Adding 900 trees for 0% gain introduces unnecessary technical debt and infrastructure costs.

### Q3: Debugging Feature Importance
**Question:** Why does the feature importance ranking change significantly between two identical runs?
**Answer:** This is due to **stochasticity** and **instability**:
**Double Randomness:** Random Forests randomly select both rows (bootstrapping) and columns (feature selection) for every split. Without a fixed `random_state`, these selections change every time.
**Model Convergence:** With only 10 trees (as seen in the snippet), the model hasn't built enough "consensus". A forest typically needs more estimators (e.g., 100+) for feature importance scores to stabilize and become reproducible.

---

## Part D: Out-of-Bag (OOB) Error

### Non-Technical Analogy
Imagine a teacher preparing a class for a final exam using a massive bank of 1,000 flashcards For every practice quiz, the teacher randomly hides 300 cards and only lets the students study the other 700 The **OOB Error** is the score the students get when they are quizzed specifically on those 300 hidden cards they never saw during study time.

### Technical Validation
**Proxy for Test Error:** OOB error is an excellent proxy for Test Error because it validates the model on "unseen" data without requiring a separate validation set.
**Discrepancies:** OOB error may differ significantly from Test Error if the dataset is so small that the "left-out" data isn't representative, or if there is a temporal shift (e.g., training on 2024 data to predict 2026 outcomes).
