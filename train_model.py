import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def train_random_forest(features_df):
    """
    (Robust Version)
    Takes a DataFrame, splits it, trains a Random Forest classifier,
    and saves the model and encoder.
    """
    print("\n--- Starting Random Forest Training ---")

    # --- 1. Prepare Features (X) and Labels (y) ---
    # Define columns to drop. These are not features for the model.
    cols_to_drop = ['time_window', 'car_model', 'attack_type']
    # Filter the list to only include columns that actually exist in the DataFrame
    existing_cols_to_drop = [col for col in cols_to_drop if col in features_df.columns]
    
    X = features_df.drop(columns=existing_cols_to_drop)
    y_text = features_df['attack_type']

    # --- 2. Encode Labels ---
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    joblib.dump(encoder, 'label_encoder.joblib')
    print(f"Labels encoded. Found {len(encoder.classes_)} classes.")

    # --- 3. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    # --- 4. Train Model ---
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Random Forest training complete.")
    
    # --- 5. Save Model ---
    model_filename = 'random_forest_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved successfully to {model_filename}")

def train_isolation_forest(features_df):
    """
    (Robust Version)
    Takes a DataFrame, trains an Isolation Forest model on normal traffic,
    and saves the trained model to a file.
    """
    print("\n--- Starting Isolation Forest Training ---")
    
    normal_df = features_df[features_df['attack_type'] == 'attack-free'].copy()
    if normal_df.empty:
        print("Error: No 'attack-free' data found for training.")
        return

    # Define columns to drop
    cols_to_drop = ['time_window', 'car_model', 'attack_type']
    existing_cols_to_drop = [col for col in cols_to_drop if col in normal_df.columns]
    
    X_train = normal_df.drop(columns=existing_cols_to_drop)
    
    print(f"Training Isolation Forest model on {len(X_train)} normal samples...")
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train)
    print("Isolation Forest training complete.")

    model_filename = 'isolation_forest_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved successfully to {model_filename}")

# --- Main execution block ---
if __name__ == '__main__':
    feature_file = 'features.csv'
    try:
        print(f"Loading feature data from '{feature_file}'...")
        # Load the data ONCE at the start
        main_features_df = pd.read_csv(feature_file)
        
        # Pass the DataFrame to each training function
        train_isolation_forest(main_features_df)
        train_random_forest(main_features_df)

    except FileNotFoundError:
        print(f"Error: '{feature_file}' not found.")
        print("Please run create_shuffled_features.py or feature_engineering.py first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")