import pandas as pd
import numpy as np

# --- 1. Define File Names ---
pheno_file = 'quantitative_traits.csv'
geno_file = '12k_ld_imputed.csv'
output_file = 'merged_rice_data.csv'  # This is the file you will give to your model

# --- 2. Load the Datasets ---
print(f"Loading phenotype data from '{pheno_file}'...")
pheno_df = pd.read_csv(pheno_file)

print(f"Loading genotype data from '{geno_file}'... (This may take a moment)")
geno_df = pd.read_csv(geno_file)

print("Data loaded.")

# --- 3. Rename the Key Column ---
# We found the common key was 'Unnamed: 0'
key_col_old = 'Unnamed: 0'
key_col_new = 'Sample_ID'  # A much better name

print(f"Renaming key column '{key_col_old}' to '{key_col_new}'...")
pheno_df = pheno_df.rename(columns={key_col_old: key_col_new})
geno_df = geno_df.rename(columns={key_col_old: key_col_new})

# --- 4. Merge the DataFrames ---
print(f"Merging dataframes on '{key_col_new}'...")
full_data_df = pd.merge(geno_df, pheno_df, on=key_col_new)

print("Merge successful. Final DataFrame shape:", full_data_df.shape)
print(full_data_df.head())

# --- 5. Save the Final Merged File ---
print(f"\nSaving final merged file to '{output_file}'...")
# index=False is crucial to avoid adding a new 'Unnamed: 0' column
full_data_df.to_csv(output_file, index=False)

print(f"\n✅ Success! Your new file is ready.")
print(f"You can now use this file for training: {output_file}")