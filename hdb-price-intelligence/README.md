# HDB Price Intelligence

A modular ML pipeline to predict Singapore HDB resale prices using Random Forest ensembles.

## Performance
| Metric | Baseline | Tuned |
| :--- | :--- | :--- |
| **Test $R^2$ Score** | 0.7875 | **0.7950** |
| **MAE (Avg Error)** | ~$43,000 | **~$41,500** |
| **Best CV $R^2$** | N/A | 0.8232 |

---

## Pipeline Breakdown

### 1. Data Ingestion
* **Source:** Data.gov.sg REST API.
* **Logic:** Authenticated headers via `x-api-key` and `.env` for secure credential management.

### 2. Feature Engineering
Transform raw strings into high-signal numerical features:
* **`remaining_lease`** $\rightarrow$ Decimal years (e.g., "95 years 7 months" to `95.58`).
* **`storey_range`** $\rightarrow$ Midpoint integer (e.g., "04 TO 06" to `5.0`).
* **`town`** $\rightarrow$ Binary flags: `is_mature` and `is_central`.
* **`resale_price`** $\rightarrow$ Binary label `price_high` for classification tasks.

### 3. Evaluation Harness
* Built a standalone module to track Regression ($R^2$, MAE, RMSE) and Classification (F1, Recall).
* Ensures all model iterations are compared against a locked 20% unseen test set.

### 4. Hyperparameter Optimization
* **Method:** `GridSearchCV` with 5-fold Cross-Validation.
* **Result:** Optimized `max_depth: 10` and `n_estimators: 200` to maximize generalization.

---

## Key Insights
* **Generalization:** Solved severe overfitting in Decision Trees (31% gap) by transitioning to Tuned Random Forests (3% gap).
* **Feature Signal:** `floor_area_sqm` is the primary driver, while `lease_rem_years` accounts for property value decay.
* **Inference Stability:** Prioritized stable Cross-Validation scores over individual test splits for more reliable deployment.

---

## Roadmap
- **Month 2:** ML Pipeline & Model Optimization (Complete)
- **Month 3:** FastAPI Deployment, Streamlit UI, and Azure Cloud Integration.