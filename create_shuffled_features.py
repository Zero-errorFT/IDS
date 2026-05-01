import pandas as pd
from feature_engineering import create_time_window_features # Assumes this is your final, fast version
from data_loader import load_dataset_from_folders
import os

def create_and_shuffle_features():
    print("--- Creating a new, shuffled feature set for dynamic simulation ---")
    
    # Use the path to your raw dataset
    dataset_path = "C:/Users/sujal/can-dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found at {dataset_path}")
        return

    # Using a smaller sample size for quicker feature generation
    raw_df = load_dataset_from_folders(dataset_path, num_rows_per_file=10000)

    if raw_df.empty:
        print("No data loaded. Exiting.")
        return
        
    features_df = create_time_window_features(raw_df)

    print("Shuffling feature windows...")
    shuffled_df = features_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save BOTH a shuffled and an unshuffled version for flexibility
    features_df.to_csv("features.csv", index=False)
    shuffled_df.to_csv("shuffled_features.csv", index=False)
    
    print("\nSuccessfully created 'features.csv' (for training) and 'shuffled_features.csv' (for simulation).")

if __name__ == '__main__':
    create_and_shuffle_features()