# File: train.py (Modified for robust error handling during email download)
import pandas as pd
import numpy as np
import pickle
import os
import imaplib
import email
import io
import re
import traceback # Import traceback for detailed error logging

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

# --- CONFIGURATION ---
SUBJECT_FILTERS = [
    'problem statement', 
    'business problem', 
    'business use case', 
    'project details', 
    'analysis the project', 
    'dataanalysis project details'
]
# --------------------------------------------------------------------


# --- DATA INGESTION FUNCTION ---

def download_dataset_from_email() -> pd.DataFrame:
    """
    Connects to the email client, searches for the latest email with a data file (CSV) 
    matching ANY of the defined subject filters, downloads, and returns it.
    """
    AGENT_EMAIL = os.environ.get("AGENT_EMAIL")
    AGENT_PASSWORD = os.environ.get("AGENT_PASSWORD")
    
    if not AGENT_EMAIL or not AGENT_PASSWORD:
        # This error should be caught by the parent main function's try/except
        raise ValueError("Email credentials (AGENT_EMAIL, AGENT_PASSWORD) not found in environment.")

    search_terms = [f'SUBJECT "{s}"' for s in SUBJECT_FILTERS]
    search_query = f'(OR {" ".join(search_terms)})'
    
    print(f"--- INGESTION: Attempting to connect to email server... (Filters: {SUBJECT_FILTERS})")

    # The crash usually happens inside this try block (network/auth failure)
    mail = imaplib.IMAP4_SSL('imap.gmail.com', 993) 
    mail.login(AGENT_EMAIL, AGENT_PASSWORD)
    mail.select('inbox')
    
    status, email_ids = mail.search(None, f'(ALL {search_query})') 
    
    if not email_ids[0]:
        raise FileNotFoundError(f"No emails found matching any of the subject filters: {SUBJECT_FILTERS}")

    latest_email_id = email_ids[0].split()[-1]
    status, msg_data = mail.fetch(latest_email_id, '(RFC822)')
    mail.store(latest_email_id, '+FLAGS', '\\Seen') 

    msg = email.message_from_bytes(msg_data[0][1])
    df = None

    for part in msg.walk():
        if part.get_content_maintype() == 'multipart': continue
        if part.get('Content-Disposition') is None: continue
        
        filename = part.get_filename()
        if filename and (filename.endswith('.csv') or filename.endswith('.txt')):
            payload = part.get_payload(decode=True)
            
            try:
                df = pd.read_csv(io.StringIO(payload.decode('utf-8')))
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(payload.decode('latin-1')))
            
            break

    mail.logout()
    if df is None:
         raise Exception(f"Email found, but no CSV or TXT attachment was detected.")
         
    print(f"--- INGESTION: Dataset '{filename}' downloaded successfully. Shape: {df.shape}")
    return df


# --- TRAINING LOGIC ---

def run_ml_pipeline(df: pd.DataFrame):
    
    df_processed = df.copy()
    target_col = df_processed.columns[-1]
    
    is_regression = np.issubdtype(df_processed[target_col].dtype, np.number) and df_processed[target_col].nunique() > 50
    
    # Preprocessing logic remains the same
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if df_processed[col].nunique() == 2:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
    
    df_processed = pd.get_dummies(df_processed, drop_first=True)

    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if is_regression:
        print("--- TRAINING: Auto-Detected Regression Task. Using DecisionTreeRegressor.")
        ModelClass = DecisionTreeRegressor
        metric_func = r2_score
        metric_name = "R-squared Score"
    else:
        print("--- TRAINING: Auto-Detected Classification Task. Using DecisionTreeClassifier.")
        ModelClass = DecisionTreeClassifier
        metric_func = accuracy_score
        metric_name = "Accuracy Score"
    
    model = make_pipeline(StandardScaler(), ModelClass(random_state=42))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if not is_regression and np.issubdtype(y_train.dtype, np.number):
         y_pred = np.round(y_pred).astype(int) 

    metric_value = metric_func(y_test, y_pred)
    
    return model, metric_value, metric_name, ModelClass.__name__

# --- Main function to execute both steps ---
def main():
    try:
        # Step 1: Ingest Data
        df = download_dataset_from_email()

        # Step 2: Run ML Pipeline
        model, metric_value, metric_name, model_name = run_ml_pipeline(df)

        # 1. Save Model Artifact (model.pkl)
        model_filename = 'model.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"--- ARTIFACTS: Model saved as {model_filename}")

        # 2. Output Metrics to Console (CRITICAL for GitHub Actions parsing)
        print(f"***METRIC_OUTPUT***: accuracy={metric_value:.4f}")
        
        print(f"Training Complete. Final {metric_name} ({model_name}): {metric_value:.4f}")
        
        # 3. Save a simple metrics file (for artifact logging)
        with open('metrics.txt', 'w') as f:
            f.write(f"Task: {'Regression' if 'Regressor' in model_name else 'Classification'}\n")
            f.write(f"Final {metric_name}: {metric_value:.4f}\n")
            f.write(f"Model: {model_name}\n")

    except Exception as e:
        # --- CRITICAL CHANGE HERE ---
        # This block ensures the full Python traceback is printed to stdout,
        # which the ci_cd.yml script will capture.
        print(f"FATAL WORKFLOW ERROR: {type(e).__name__}: {str(e)}")
        print("--- FULL TRACEBACK ---")
        traceback.print_exc()
        # The script exits with status 1, which the ci_cd.yml will detect as a failure.
        exit(1)


if __name__ == "__main__":
    main()
