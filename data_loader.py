import pandas as pd
import os

def load_dataset_from_folders(main_directory, num_rows_per_file=50000):
    """
    Walks through a nested directory, loads a specified number of rows from each 
    CSV file, labels them, and combines them into a single DataFrame.

    Args:
        main_directory (str): The path to the main folder containing subfolders for each car.
        num_rows_per_file (int): The number of rows to load from the start of each CSV file.

    Returns:
        pandas.DataFrame: A single DataFrame containing the sampled and standardized data.
    """
    all_dataframes = []
    
    print(f"Loading {num_rows_per_file} rows from each CSV file...")
    
    for root, _, files in os.walk(main_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                try:
                    relative_path = os.path.relpath(root, main_directory)
                    path_parts = relative_path.split(os.sep)
                    
                    if len(path_parts) >= 2:
                        car_model = path_parts[0]
                        attack_type = path_parts[1]
                        
                        temp_df = pd.read_csv(file_path, nrows=num_rows_per_file)
                        
                        temp_df['car_model'] = car_model
                        temp_df['attack_type'] = attack_type
                        
                        all_dataframes.append(temp_df)
                        print(f"Loaded and labeled: {car_model} -> {attack_type} -> {file}")
                    else:
                        print(f"Skipping file in unexpected location: {file_path}")

                except Exception as e:
                    print(f"Could not load file {file_path}. Error: {e}")
                    
    if not all_dataframes:
        print("No CSV files found.")
        return pd.DataFrame()

    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # --- NEW SECTION: Standardize Column Names ---
    # This ensures consistency for all subsequent scripts.
    rename_dict = {
        'timestamp': 'Timestamp',
        'arbitration_id': 'CAN_ID',
        'data_field': 'Payload',
        # Add other potential renames here if necessary
    }
    master_df = master_df.rename(columns=rename_dict)
    print("\nStandardized column names.")
    
    print(f"\nSuccessfully created a scaled-down master DataFrame with {len(master_df)} total rows.")
    return master_df

# --- Example Usage ---
if __name__ == '__main__':
    data_path = '.' 
    
    can_df = load_dataset_from_folders(data_path, num_rows_per_file=50000)
    
    if not can_df.empty:
        print("\n--- Scaled-Down Master DataFrame Info ---")
        print(can_df.info()) # Should now show the new column names
        
        print("\n--- Example Labels ---")
        print(can_df[['car_model', 'attack_type']].drop_duplicates())