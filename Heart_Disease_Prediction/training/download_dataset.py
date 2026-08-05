import os
import shutil
from tkinter import Tk, filedialog
from kaggle.api.kaggle_api_extended import KaggleApi

# =====================================================
# Select kaggle.json
# =====================================================

print("Select your kaggle.json file")

root = Tk()
root.withdraw()

json_path = filedialog.askopenfilename(
    title="Select kaggle.json",
    filetypes=[("JSON Files", "*.json")]
)

if not json_path:
    raise Exception("No kaggle.json selected.")

# =====================================================
# Copy kaggle.json
# =====================================================

kaggle_folder = os.path.join(os.path.expanduser("~"), ".kaggle")
os.makedirs(kaggle_folder, exist_ok=True)

destination = os.path.join(kaggle_folder, "kaggle.json")

shutil.copy(json_path, destination)

try:
    os.chmod(destination, 0o600)
except Exception:
    pass

print("✓ kaggle.json configured successfully.")

# =====================================================
# Authenticate
# =====================================================

api = KaggleApi()
api.authenticate()

print("✓ Kaggle authenticated.")

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_FOLDER = os.path.join(PROJECT_ROOT, "dataset")

os.makedirs(DATASET_FOLDER, exist_ok=True)

# =====================================================
# Dataset
# =====================================================

DATASET = "rishidamarla/heart-disease-prediction"

print("\nDownloading dataset...\n")

api.dataset_download_files(
    dataset=DATASET,
    path=DATASET_FOLDER,
    unzip=True
)

print("\n===================================")
print("Dataset Download Completed")
print("===================================")
print(f"Saved to : {DATASET_FOLDER}")