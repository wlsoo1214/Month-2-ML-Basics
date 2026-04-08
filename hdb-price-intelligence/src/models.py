# src/models.py

# separate portion of data into test and train

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from evaluator import MLEvaluator  # Importing your harness
from sklearn.tree import DecisionTreeRegressor

def check_overfitting(X_train, X_test, y_train, y_test):
    print("\n--- Overfitting Stress Test: Decision Tree (No Limits) ---")
    # Decision Tree with no max_depth will try to memorize every single row
    overfit_model = DecisionTreeRegressor(max_depth=None)
    overfit_model.fit(X_train, y_train)
    
    train_score = overfit_model.score(X_train, y_train)
    test_score = overfit_model.score(X_test, y_test)
    
    print(f"Training R2: {train_score:.4f} (The 'Memorization' score)")
    print(f"Testing R2:  {test_score:.4f} (The 'Real World' score)")
    print(f"Gap: {train_score - test_score:.4f} <--- If this is large, you are overfitted!")

def run_model_pipeline():
    # 1. Load the data
    df = pd.read_csv('data/processed_hdb_data.csv')

    # 2. Define Features (X) and Targets (y)
    # Updated to match YOUR specific column names
    features = ['floor_area_sqm', 'lease_rem_years', 'storey_mid', 'is_mature', 'is_central']
    X = df[features]

    y_reg = df['resale_price']    # Target for Regression
    y_clf = df['price_high']      # Target for Classification (matches your list)

    # 3. SPLIT: Fix the 'test_size' parameter
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    print(f"Training on {len(X_train)} rows. Testing on {len(X_test)} rows.")

    # --- TASK A: REGRESSION (Predicting Price) ---
    print("\n--- Training Random Forest Regressor ---")
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_reg_train)
    
    reg_preds = reg_model.predict(X_test)
    MLEvaluator.evaluate_regression(y_reg_test, reg_preds, "HDB Price Regressor")

    # --- TASK B: CLASSIFICATION (Predicting High Value) ---
    print("\n--- Training Logistic Regression Classifier ---")
    clf_model = LogisticRegression(max_iter=1000)
    clf_model.fit(X_train, y_clf_train)
    
    clf_preds = clf_model.predict(X_test)
    MLEvaluator.evaluate_classification(y_clf_test, clf_preds, "High Value Classifier")

    # --- TASK C: OVERFITTING STRESS TEST ---
    check_overfitting(X_train, X_test, y_reg_train, y_reg_test)

if __name__ == "__main__":
    run_model_pipeline()
    