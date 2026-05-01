# In feature_engineering.py
import pandas as pd
import numpy as np
from scipy.stats import entropy
from data_loader import load_dataset_from_folders 

# (Helper functions hex_to_int_list and calculate_entropy remain the same)
def hex_to_int_list(hex_str):
    # This can be a bottleneck, but we'll keep it for now as the groupby is the main issue.
    # A faster C-extension or numba could optimize this further if needed.
    return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

def calculate_entropy(byte_series):
    # Now takes a pandas Series of bytes
    if byte_series.empty:
        return 0
    counts = np.bincount(byte_series.astype(int), minlength=256)
    return entropy(counts, base=2)

def create_time_window_features(df, window_size=1.0):
    """
    (Optimized Version)
    Engineers advanced time-window features from the raw CAN data DataFrame.
    """
    print(f"Creating advanced features with a {window_size} second time window (Optimized)...")
    
    df['time_window'] = (df['Timestamp'] // window_size).astype(int)

    # --- Feature 1: Message Frequency (already fast) ---
    freq_df = df.pivot_table(index='time_window', columns='CAN_ID', values='Timestamp', aggfunc='count').fillna(0)
    freq_df.columns = [f'{col}_freq' for col in freq_df.columns]

    # --- Data Cleaning (as before) ---
    df.dropna(subset=['Payload'], inplace=True)
    df['Payload'] = df['Payload'].astype(str).str.replace(r'[^0-9a-fA-F]', '', regex=True)
    
    # --- OPTIMIZED FEATURE CALCULATION ---
    # 1. Create the integer list from hex payloads
    df['Payload_Int'] = df['Payload'].apply(hex_to_int_list)
    
    # 2. Explode the DataFrame so each byte has its own row. This is key.
    exploded_df = df[['time_window', 'CAN_ID', 'Payload_Int']].explode('Payload_Int')
    exploded_df.rename(columns={'Payload_Int': 'byte'}, inplace=True)
    exploded_df.dropna(subset=['byte'], inplace=True) # Drop rows if payload was empty
    exploded_df['byte'] = pd.to_numeric(exploded_df['byte'])

    # 3. Group the new, long-form DataFrame
    grouped_exploded = exploded_df.groupby(['time_window', 'CAN_ID'])['byte']

    # 4. Use fast, built-in aggregators or cleaner apply
    print("Calculating mean and entropy...")
    mean_df = grouped_exploded.mean().unstack(fill_value=0)
    entropy_df = grouped_exploded.apply(calculate_entropy).unstack(fill_value=0)

    # 5. Rename columns for clarity
    mean_df.columns = [f'{col}_mean' for col in mean_df.columns]
    entropy_df.columns = [f'{col}_entropy' for col in entropy_df.columns]
    
    # --- Combine all features ---
    print("Combining all features...")
    features_df = freq_df.join(mean_df).join(entropy_df)
    
    # --- Carry over labels ---
    labels_df = df.groupby('time_window')[['car_model', 'attack_type']].first()
    
    final_df = features_df.join(labels_df).fillna(0)
    
    print(f"Successfully created an advanced feature DataFrame with {len(final_df)} time windows.")
    return final_df

# (The __main__ block remains the same)
if __name__ == '__main__':
    # Make sure to use the absolute path to your dataset
    master_can_df = load_dataset_from_folders("C:/Users/sujal/can-dataset") 
    
    if not master_can_df.empty:
        feature_df = create_time_window_features(master_can_df, window_size=1.0)
        
        print("\n--- Feature DataFrame Info ---")
        print(f"Shape of feature DataFrame: {feature_df.shape}")
        
        print("\n--- Example Features and Labels (last 5 rows) ---")
        print(feature_df.tail())

        feature_df.to_csv('features.csv')
        print("\nFeature DataFrame saved to features.csv")