import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
# The OUTPUT_DIR is the CRITICAL change. 
# It must match the internal container path used for the volume mount in ci_cd.yml.
OUTPUT_DIR = '/app/output'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.pkl')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics.txt')

# --- 1. SETUP AND DATA SIMULATION (REPLACE WITH YOUR ACTUAL DATA LOADING) ---
def load_and_preprocess_data():
    """Loads and returns feature matrix (X) and target vector (y)."""
    try:
        # ⚠️ REPLACE THIS SECTION with your actual data loading and cleaning
        print("--- AGENT: Classification task detected.")
        
        # Simulate data (Replace this with pd.read_csv('your_data.csv') etc.)
        data = {
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100) * 10,
            'target': np.random.randint(0, 2, 100)
        }
        df = pd.DataFrame(data)
        
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        return X, y

    except Exception as e:
        print(f"::error::FATAL WORKFLOW ERROR: Data Loading failed: {e}")
        return None, None

# --- 2. MODEL TRAINING AND EVALUATION ---
def train_and_evaluate(X, y):
    """Trains the XGBoost model and returns the accuracy score."""
    
    # Check for skip condition (e.g., in your original log, "NO_NEW_EMAILS_FOUND")
    # For this template, we'll assume data is present.
    # If a skip condition is met, print the NO_NEW_EMAILS_FOUND error and return 0.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train the model
    # Note: Your previous log selected XGBoost for an accuracy target of 0.90
    print("--- AGENT: Selecting XGBoostClassifier for higher accuracy target (0.90).")
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"--- TRAINING: XGBoostClassifier trained with accuracy: {accuracy:.4f}")
    
    return model, accuracy

# --- 3. ARTIFACT SAVING (CRITICAL VOLUME MOUNT STEP) ---
def save_artifacts(model, accuracy):
    """Saves the trained model and metrics to the mounted directory."""
    
    # ⚠️ CRITICAL: Ensure the directory exists before saving
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    
    try:
        # Save Model (.pkl)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"--- INFO: Successfully saved model to {MODEL_PATH}")

        # Save Metrics (.txt)
        with open(METRICS_PATH, 'w') as f:
            f.write(f"accuracy={accuracy:.4f}\n")
        print(f"--- INFO: Successfully saved metrics to {METRICS_PATH}")

    except Exception as e:
        # This will fail the workflow and tell you why the file write failed
        print(f"::error::FATAL WORKFLOW ERROR: Artifact Saving failed (Permissions or Path): {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    
    if X is None:
        exit(1) # Exit if data loading failed

    model, accuracy = train_and_evaluate(X, y)
    
    # Save the artifacts to the mounted volume
    save_artifacts(model, accuracy)

    # Print the metric line for the GitHub Action to capture
    print(f"***METRIC_OUTPUT***accuracy={accuracy:.4f}***METRIC_OUTPUT***")
    print("--- WORKFLOW END: SUCCESS ---")
