import requests
import pandas as pd
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

def get_headers():
    """
    Handles API key retrieval and header construction.

    Returns:
        dict: The headers to be used for the API request.
    """
    api_key = os.getenv("DATA_GOV_API_KEY")
    if not api_key:
        raise ValueError("DATA_GOV_API_KEY not found in .env file")
    
    return {"x-api-key": api_key}

def _base_fetch(limit):
    """
    Internal helper to handle the actual HTTP request logic.
    Send request to API and return the response (with try-except block)

    Args:
        limit (int): The number of records to fetch.
    
    Returns:
        dict: The response from the API.
    """
    resource_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={resource_id}&limit={limit}"
    
    try:
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

def preview_hdb_data(limit=2):
    """
    Function 1: Preview the raw JSON structure.

    Args:
        limit (int): The number of records to fetch.
    
    Returns:
        dict: The response from the API.
    """
    print(f"--- PREVIEWING RAW JSON (Limit: {limit}) ---")
    data = _base_fetch(limit)
    
    if data and data.get('success'):
        # This shows the 'tree' structure you wanted to see
        print(json.dumps(data, indent=4))
        print("--- END OF PREVIEW ---\n")
    else:
        print("Failed to retrieve preview data.")

def ingest_hdb_data(limit=5000):
    """
    Function 2: Real ingestion. Converts to DataFrame and saves to CSV.
    """
    print(f"--- STARTING INGESTION (Limit: {limit}) ---")
    data = _base_fetch(limit)
    
    if data and data.get('success'):
        # Map the specific nested keys fixed by the API
        records = data['result']['records']
        df = pd.DataFrame(records)
        
        # Setup paths (Saving to /data folder in project root)
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root /'src' / 'data'
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = output_dir / 'raw_hdb_data.csv'
        df.to_csv(file_path, index=False)
        
        print(f"✅ Success! {len(df)} rows saved to: {file_path}")
        return df
    else:
        print("Ingestion failed.")
        return None

if __name__ == "__main__":
    # 1. Run the data preview
    preview_hdb_data(limit=2)
    
    # 2. Run the ingestion to save the CSV
    ingest_hdb_data(limit=5000)