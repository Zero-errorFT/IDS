import pandas as pd
import os

def create_definitive_test_log(base_path, normal_file, attack_file, output_file, normal_rows=30000, attack_rows=30000):
    """
    (Corrected Types)
    Creates a single, continuous log file by combining a known normal period
    with a known attack period, using a robust CSV parsing method.
    """
    print("--- Creating a definitive test log ---")
    
    try:
        # --- 1. Load Normal Traffic ---
        print(f"Loading {normal_rows} rows from normal file...")
        normal_file_path = os.path.join(base_path, normal_file)
        df_normal = pd.read_csv(normal_file_path, nrows=normal_rows)

        # --- 2. Load Attack Traffic ---
        print(f"Loading {attack_rows} rows from attack file...")
        attack_file_path = os.path.join(base_path, attack_file)
        header = pd.read_csv(attack_file_path, nrows=0).columns.tolist()
        num_lines = sum(1 for l in open(attack_file_path))
        skip_rows = max(1, (num_lines // 2) - (attack_rows // 2))
        df_attack = pd.read_csv(
            attack_file_path, 
            skiprows=range(1, skip_rows), 
            nrows=attack_rows, 
            header=None
        )
        df_attack.columns = header
        
        # --- 3. Combine and Fix Timestamps ---
        print("Combining and creating a continuous timeline...")
        df_normal = df_normal.rename(columns={'timestamp': 'Timestamp', 'arbitration_id': 'CAN_ID', 'data_field': 'Payload'})
        df_attack = df_attack.rename(columns={'timestamp': 'Timestamp', 'arbitration_id': 'CAN_ID', 'data_field': 'Payload'})

        # --- NEW SECTION TO FIX THE ERROR ---
        # Force the Timestamp columns to be a numeric type, coercing any errors
        df_normal['Timestamp'] = pd.to_numeric(df_normal['Timestamp'], errors='coerce')
        df_attack['Timestamp'] = pd.to_numeric(df_attack['Timestamp'], errors='coerce')
        # Drop any rows where the conversion failed
        df_normal.dropna(subset=['Timestamp'], inplace=True)
        df_attack.dropna(subset=['Timestamp'], inplace=True)
        
        last_normal_timestamp = df_normal['Timestamp'].iloc[-1]
        
        # This subtraction will now work correctly
        df_attack['Timestamp'] = df_attack['Timestamp'] - df_attack['Timestamp'].iloc[0] + last_normal_timestamp + 0.01
        
        final_df = pd.concat([df_normal, df_attack], ignore_index=True)
        final_df = final_df[['Timestamp', 'CAN_ID', 'Payload']]
        
        final_df.to_csv(output_file, index=False)
        duration = final_df['Timestamp'].max() - final_df['Timestamp'].min()
        print(f"\nSuccessfully created '{output_file}' with {len(final_df)} rows.")
        print(f"Estimated simulation duration: {int(duration)} seconds.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the file paths at the bottom of the script are correct.")


if __name__ == '__main__':
    base_data_path = "/mnt/c/Users/sujal/can-dataset"
    normal_log = "2011-chevrolet-impala/attack-free/attack-free-1.csv"
    attack_log = "2011-chevrolet-impala/DoS-attacks/DoS-1.csv"
    output_log_file = "test_simulation.csv"

    create_definitive_test_log(base_data_path, normal_log, attack_log, output_log_file)