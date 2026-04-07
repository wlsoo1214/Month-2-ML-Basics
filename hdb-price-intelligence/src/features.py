import pandas as pd
import numpy as np 

def engineer_features(df):
    """
    Turn raw strings into math signals
    """
    print("Engineering features...")
    processed_df = df.copy()

    # 1. Target Variable (Labels)
    # Convert 'resale_price' to numeric
    processed_df['price'] = pd.to_numeric(processed_df['resale_price'], errors='coerce')
    
    # Classification Target: High/Low
    # Top 25% = High (1), Bottom 75% = Low (0)
    threshold = processed_df['price'].quantile(0.75)
    processed_df['price_high'] = (processed_df['price'] >= threshold).astype(int)
    
    # Feature Engineering
    # FEATURE 1: Remaining lease
    def parse_lease(lease_str):
        if pd.isna(lease_str):
            return 0
        parts = str(lease_str).split() 
        try:
            years = int(parts[0])
            # Check if 'months' exists (index 2)
            months = int(parts[2]) if len(parts) > 2 else 0
            return years + months / 12
        except (IndexError, ValueError):
            return 0

    processed_df['lease_rem_years'] = processed_df['remaining_lease'].apply(parse_lease)

    # FEATURE 2: Storey level (midpoint)
    # FIX: Changed storage_str to storey_str to match function argument
    def parse_storey(storey_str):
        if pd.isna(storey_str):
            return 0
        try:
            low, high = storey_str.split(" TO ")
            return (int(low) + int(high)) / 2
        except:
            return 0
            
    processed_df['storey_mid'] = processed_df['storey_range'].apply(parse_storey)

    # FEATURE 3: Is Mature Estate?
    mature_towns = ['BISHAN', 'BUKIT MERAH', 'GEYLANG', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'TAMPINES', 'TOA PAYOH']
    processed_df['is_mature'] = processed_df['town'].isin(mature_towns).astype(int)

    # FEATURE 4: Distance to city
    central_towns = ['CENTRAL AREA', 'BUKIT MERAH', 'QUEENSTOWN', 'KALLANG/WHAMPOA']    
    processed_df['is_central'] = processed_df['town'].isin(central_towns).astype(int)

    # FEATURE 5: Price per SQM
    # FIX: Using the numeric 'price' column instead of the raw string column
    processed_df['floor_area_sqm'] = pd.to_numeric(processed_df['floor_area_sqm'], errors='coerce')
    processed_df['price_per_sqm'] = processed_df['price'] / processed_df['floor_area_sqm']

    return processed_df

if __name__ == "__main__":
    # Ensure you are running this from within the 'src' folder
    try:
        # Load raw data
        raw_df = pd.read_csv("data/raw_hdb_data.csv")
        
        # Engineer features
        processed_df = engineer_features(raw_df)

        # Save processed data
        processed_df.to_csv("data/processed_hdb_data.csv", index=False)
        print("Feature engineering complete. Check data/processed_hdb_data.csv")
        
    except FileNotFoundError:
        print("Error: Could not find 'data/raw_hdb_data.csv'.")
        print("Make sure you are in the 'src' directory before running the script.")