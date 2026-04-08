## Model Performance (Month 2)

I implemented a Random Forest Regressor to predict Singapore HDB resale prices. 
The pipeline includes data ingestion from Data.gov.sg, feature engineering, and hyperparameter tuning.

| Metric | Baseline (RF) | Tuned (RF) |
| :--- | :--- | :--- |
| **Test R2 Score** | 0.7875 | 0.7950 |
| **Mean Absolute Error** | ~$43,000 | ~$41,500 |
| **Key Features** | Area, Lease, Maturity | Area, Lease, Maturity |

### Lessons Learned
- **Overfitting:** A raw Decision Tree achieved 0.99 training R2 but only 0.68 testing R2. 
- **Regularization:** Limiting `max_depth` to 10 significantly improved generalization.
- **Cross-Validation:** 5-fold CV provided a more reliable estimate of model performance than a single split.