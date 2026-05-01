import pandas as pd
from feature_engineering import create_time_window_features
from data_loader import load_dataset_from_folders
import os

def create_and_shuffle_features():
    """
    Loads all data, creates features, shuffles them, and saves the result.
    This creates a more realistic dataset for simulations.
    """
    print("--- Creating a new, shuffled feature set for dynamic simulation ---")
    
    # Define the path to your dataset
    dataset_path = "C:/Users/sujal/can-dataset" # Use your actual path
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found at {dataset_path}")
        return

    # 1. Load the raw data (using a smaller sample for speed if needed)
    raw_df = load_dataset_from_folders(dataset_path, num_rows_per_file=10000) # Using 10k for a faster, mixed sample

    if raw_df.empty:
        print("No data loaded. Exiting.")
        return
        
    # 2. Create the time-window features as before
    features_df = create_time_window_features(raw_df)

    # 3. Shuffle the DataFrame
    # This is the key step to mix normal and attack windows
    print("Shuffling feature windows...")
    shuffled_df = features_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. Save the new file
    output_filename = "shuffled_features.csv"
    shuffled_df.to_csv(output_filename, index=False)
    
    print(f"\nSuccessfully created '{output_filename}' with {len(shuffled_df)} shuffled time windows.")
    print("This file should now be used for the real-time simulation.")

if __name__ == '__main__':
    create_and_shuffle_features()
