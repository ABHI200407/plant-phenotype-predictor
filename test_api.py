import requests
import pandas as pd
import json
import time

# --- CONFIG ---
BASE_URL = "http://localhost:5000"
DATA_FILE = "merged_rice_data.csv"

def get_real_data_sample():
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    df.fillna(0, inplace=True)
    
    # Define columns to drop (Must match training exactly)
    drop_cols = ['Sample_ID', 'GRWT100', 'GRLT', 'GRWD', 'HDG_80HEAD', 
                 'LIGLT', 'LLT', 'LWD', 'PLT_POST', 'SDHT', 'CUNO_REPRO']
    existing_drops = [c for c in drop_cols if c in df.columns]
    
    # Get the actual target value for comparison
    actual_value = df.iloc[0]['GRWT100']
    
    # Get features for the first row
    features_df = df.drop(existing_drops, axis=1)
    input_data = features_df.iloc[0].tolist() 
    
    return input_data, actual_value

def run_test():
    # 1. GET DATA
    try:
        features, actual = get_real_data_sample()
        print(f"Loaded sample data. Input features: {len(features)}")
        print(f"Actual Value to expect: {actual}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. REGISTER
    print("\n1. Registering User...")
    auth_data = {"username": "testuser", "password": "password123"}
    try:
        requests.post(f"{BASE_URL}/register", json=auth_data)
    except Exception as e:
        print(f"Connection error: {e}")
        return

    # 3. LOGIN
    print("2. Logging in...")
    response = requests.post(f"{BASE_URL}/login", json=auth_data)
    if response.status_code != 200:
        print("Login failed")
        return
    
    token = response.json()['access_token']
    headers = {"Authorization": f"Bearer {token}"}
    print("Login successful. Token received.")

    # 4. PREDICT
    print("\n3. Sending Prediction Request...")
    payload = {"features": features}
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=payload, headers=headers)
    
    if response.status_code == 200:
        task_info = response.json()
        print(f"Task Submitted. Task ID: {task_info['task_id']}")
        print(f"Time to submit: {time.time() - start_time:.2f}s")
        print("\nCHECK YOUR DOCKER TERMINAL NOW.")
        print("You should see the Worker process this task and print the result.")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    run_test()