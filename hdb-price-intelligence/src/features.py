import pandas as pdb
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
    # price_high = 1 if price >= threshold else 0
    processed_df['price_high'] = (processed_df['price'] >= threshold).astype(int)
    
    # Feature Engineering
    # FEATURE 1: Remaining lease
    # 95 years 03 months -> 95.25
    def parse_lease(lease_str):
        parts = lease_str.split() # ['95', 'years', '03', 'months']
        years = int(parts[0])
        months = int(parts[2])
        return years + months / 12
    processed_df['lease_rem_years'] = processed_df['lease_remaining'].apply(parse_lease)
    
    # FEATURE 2: Storage level (midpoint)
    # Convert "04 TO 06" -> 5.0
    def parse_storage(storage_str):
        low, high = storage_str.split(" TO ")
        return (int(low) + int(high)) / 2
    processed_df['storage_mid'] = processed_df['storage_area'].apply(parse_storage) 

    # FEATURE 3: Is Mature Estate? (Domain Knowledge)        
    mature_towns = ['BISHAN', 'BUKIT MERAH', 'GEYLANG', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'TAMPINES', 'TOA PAYOH']
    # .isin returns boolean
    # .astype(int) converts boolean to int (True -> 1, False -> 0)
    processed_df['is_mature'] = processed_df['town'].isin(mature_towns).astype(int)

    # FEATURE 4: Distance to city
    # 1 = central/south, 0 otherwise
    central_towns = ['CENTRAL AREA', 'BUKIT MERAH', 'QUEENSTOWN', 'KALLANG/WHAMPOA']    
    processed_df['is_central'] = processed_df['town'].isin(central_towns).astype(int)

    # FEATURE 5: Price per SQM
    # price / floor_area
    processed_df['floor_area_sqm'] = pd.to_numeric(processed_df['floor_area_sqm'])
    processed_df['price_per_sqm'] = processed_df['resale_price'] / processed_df['floor_area_sqm']

    return processed_df

if __name__ == "__main__":
    # Load raw data
    raw_df = pd.read_csv("data/raw_hdb_data.csv")
    
    # Engineer features
    processed_df = engineer_features(raw_df)

    # Save processed data
    processed_df.to_csv("data/processed_hdb_data.csv", index=False)
    print("Feature engineering complete. Check data/processed_hdb_data.csv")