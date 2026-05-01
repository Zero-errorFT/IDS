import pandas as pd
import joblib
import time

# --- NEW: Import the trusted function from your other script ---
from feature_engineering import create_time_window_features

def run_definitive_simulation(log_file_path):
    """
    (Definitive Version)
    Processes a CAN log by calling the trusted feature engineering function,
    ensuring consistency between training and simulation.
    """
    print("--- Starting Definitive Simulation & Detection ---")
    
    # --- 1. Load Model and Utilities ---
    try:
        rf_model = joblib.load('random_forest_model.joblib')
        encoder = joblib.load('label_encoder.joblib')
        print("Models and utilities loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: A required model file was not found: {e}")
        return

    # --- 2. Load the Entire Log File ---
    try:
        log_df = pd.read_csv(log_file_path)
        log_df = log_df.rename(columns={'timestamp': 'Timestamp', 'arbitration_id': 'CAN_ID', 'data_field': 'Payload'})
        print(f"Loaded {len(log_df)} messages from {log_file_path}.")
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return
        
    # --- 3. Generate All Features at Once Using the Trusted Function ---
    # This ensures features are 100% consistent with the training data.
    print("Generating features for the entire simulation log...")
    feature_df = create_time_window_features(log_df, window_size=1.0)
    
    if feature_df.empty:
        print("Feature engineering resulted in an empty DataFrame. Exiting.")
        return
    
    # Separate features from labels for prediction
    X_to_predict = feature_df.drop(columns=['time_window', 'car_model', 'attack_type'])
    
    # --- 4. Make All Predictions at Once ---
    print("Making predictions on all time windows...")
    predictions_encoded = rf_model.predict(X_to_predict)
    predictions_decoded = encoder.inverse_transform(predictions_encoded)
    
    # Add predictions to our feature DataFrame for easy display
    feature_df['prediction'] = predictions_decoded

    print("\n--- Live Traffic Analysis (Fast-Forward) ---")
    all_predictions = []
    
    # --- 5. Loop Through Results and Print ---
    for index, row in feature_df.iterrows():
        prediction_text = row['prediction']
        window_id = row['time_window']
        
        all_predictions.append(prediction_text)
        status_color = "\033[91m" if "attack" in prediction_text and "free" not in prediction_text else "\033[92m"
        reset_color = "\033[0m"
        print(f"[Window ID: {window_id}] Status: {status_color}{prediction_text.upper()}{reset_color}")
        time.sleep(0.05)
            
    print("\n--- Simulation Complete ---")

    print("\n--- Final Summary of Detected Events ---")
    summary = pd.Series(all_predictions).value_counts()
    print(summary)

if __name__ == '__main__':
    # This script will now run on our new, perfectly structured test file
    simulation_file = "test_simulation.csv"
    run_definitive_simulation(log_file_path=simulation_file)