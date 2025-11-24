import pandas as pd
import numpy as np

# 1. Read your reference data to get the exact columns
try:
    print("Reading merged_rice_data.csv to learn column names...")
    ref_df = pd.read_csv('merged_rice_data.csv')
    
    # 2. Create 5 rows of random data
    print("Generating random dummy data...")
    dummy_data = pd.DataFrame(np.random.rand(5, len(ref_df.columns)), columns=ref_df.columns)
    
    # 3. Fix the ID column so it looks like a real sample name
    # (The code drops 'Sample_ID', but it's good to have it for the output CSV)
    if 'Sample_ID' in dummy_data.columns:
        dummy_data['Sample_ID'] = ['Test_Sample_01', 'Test_Sample_02', 'Test_Sample_03', 'Test_Sample_04', 'Test_Sample_05']

    # 4. Save to a new file
    output_name = 'dummy_test.csv'
    dummy_data.to_csv(output_name, index=False)
    print(f" Success! Created '{output_name}' with 5 test rows.")

except FileNotFoundError:
    print("Error: Could not find 'merged_rice_data.csv'. Run this where your data is.")