import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib # To save the best model

"""
Use GridSearchCV 

"""

def tune_hyperparameters():
    # 1. Load data
    df = pd.read_csv("data/processed_hdb_data.csv")

    # 2. Define Features and Target
    features = ['floor_area_sqm', 'lease_rem_years', 'storey_mid', 'is_mature', 'is_central']
    X = df[features]
    y = df['resale_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define the search space (The menu) 
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [10, 20, None],     # Max depth of trees
        'min_samples_split': [2, 5, 10], # Minimum n of samples required to split a node
        'max_features': ['sqrt', 'log2'] # Max n of features to consider when splitting a node
    }
    
    print("Starting Grid Search with 5-Fold Cross Validation...")
    print(f"Testing {2 * 3 * 3 * 2 * 5} total model combinations. This may take a minute...")

    # 4. Intialize grid search
    # cv=5 means 5-fold cross-validation
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="r2", # R-squared, for regression
        verbose=1)

# 5. Execute the Search
    grid_search.fit(X_train, y_train)

    # 6. Results
    print("\n--- Tuning Results ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV R2 Score: {grid_search.best_score_:.4f}")

    # 7. Final Test Evaluation
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"Final Test R2 Score (Unseen Data): {test_score:.4f}")

    # Save the model for Month 3 (Deployment)
    joblib.dump(best_model, 'data/best_hdb_model.pkl')
    print("Model saved to data/best_hdb_model.pkl")

if __name__ == "__main__":
    tune_hyperparameters()