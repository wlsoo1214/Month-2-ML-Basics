# HDB Price Intelligence: End-to-End ML Pipeline

A data-driven system designed to predict Singapore HDB resale prices using machine learning. This project moves beyond simple modeling to implement a full engineering lifecycle—from automated ingestion to hyperparameter optimization.

---

## Model Performance (Month 2 Milestone)

By transitioning from a baseline Random Forest to a tuned ensemble, the system achieved a significant increase in inference reliability.

| Metric | Baseline (Random Forest) | Optimized (Tuned RF) |
| :--- | :--- | :--- |
| **Test $R^2$ Score** | 0.7875 | **0.7950** |
| **Mean Absolute Error (MAE)** | ~$43,000 | **~$41,500** |
| **CV Stability ($R^2$)** | N/A | 0.8232 |

---

## Project Lifecycle & Architecture

The system is built as a modular pipeline to ensure reproducibility and scalability.

### 1. Automated Data Ingestion
* **Goal:** Establish a secure, reliable connection to the source of truth.
* **Engineering Logic:** Developed a robust ingestion script to interface with the Data.gov.sg REST API. Implemented secure credential management via environment variables (`.env`) and handled rate-limiting through authenticated headers.

### 2. Feature Engineering & Vectorization
* **Goal:** Transform raw, human-readable strings into high-signal numerical features.
* **Inference Logic:** Utilized vectorized operations to parse complex strings (e.g., remaining lease and storey ranges) into continuous floats. Added domain-specific features such as `is_mature` and `is_central` to capture geographical price premiums.

### 3. Evaluation & Validation Harness
* **Goal:** Create a standardized "Judge" to prevent biased reporting.
* **Implementation:** Built a standalone evaluation module to track both Regression (RMSE, MAE, $R^2$) and Classification (F1-Score, Recall) metrics. This ensures every model iteration is measured against the same "Real World" yardstick.

### 4. Optimization & Hyperparameter Tuning
* **Goal:** Minimize the "Generalization Gap" between training and testing.
* **Strategy:** Executed a `GridSearchCV` with 5-fold cross-validation. By constraining `max_depth` and optimizing `n_estimators`, the model moved from "memorizing" noise to "learning" market trends.

---

## Engineering Insights & Findings

* **Identifying Overfitting:** Initial "Stress Tests" using unconstrained Decision Trees revealed a critical generalization gap (0.99 Training vs. 0.68 Testing). This highlighted the necessity of ensemble methods like Random Forests to handle high-variance data.
* **Feature Signal:** Data analysis confirms that `floor_area_sqm` remains the primary driver of absolute value, while `lease_rem_years` provides the necessary decay signal for older properties.
* **Stability over Accuracy:** Prioritized the Cross-Validation (CV) score over individual test splits. A stable 0.79 $R^2$ is more valuable for deployment than a "lucky" 0.81 split that fails on diverse data.

---

## Roadmap
- **Month 2:** ML Foundations & Pipeline Construction (Current)
- **Month 3:** Productionization via FastAPI, Streamlit Dashboarding, and Azure Cloud Deployment.