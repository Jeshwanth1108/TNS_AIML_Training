"""
=========================================================
Heart Disease Dataset Verification
=========================================================
"""

import os
import pandas as pd

# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FOLDER = os.path.join(PROJECT_ROOT, "dataset")

# ---------------------------------------------------------
# Locate CSV File
# ---------------------------------------------------------

csv_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV file found in dataset folder.")

csv_path = os.path.join(DATASET_FOLDER, csv_files[0])

print("=" * 60)
print("Dataset Found")
print("=" * 60)
print(csv_path)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

df = pd.read_csv(csv_path)

print("\nDataset Loaded Successfully\n")

print("=" * 60)
print("Shape")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("Columns")
print("=" * 60)

for col in df.columns:
    print(col)

print("\n")

print("=" * 60)
print("Data Types")
print("=" * 60)

print(df.dtypes)

print("\n")

print("=" * 60)
print("Missing Values")
print("=" * 60)

print(df.isnull().sum())

print("\n")

print("=" * 60)
print("Duplicate Rows")
print("=" * 60)

print(df.duplicated().sum())

print("\n")

print("=" * 60)
print("Target Distribution")
print("=" * 60)

# Change "target" if your dataset uses a different target column name.
target_col = "target"

if target_col in df.columns:
    print(df[target_col].value_counts())
else:
    print("Target column not found.")

print("\n")

print("=" * 60)
print("First Five Rows")
print("=" * 60)

print(df.head())