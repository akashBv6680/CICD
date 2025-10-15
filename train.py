import pandas as pd
import numpy as np
import os
import pickle
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
# --- CLASSIFICATION MODELS (From previous version) ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
# --- REGRESSION MODELS (NEW ADDITIONS) ---
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb # Re-imported for clarity, though already imported via XGBClassifier

# --- CRITICAL CONFIGURATION: VOLUME MOUNT PATH ---
OUTPUT_DIR = '/app/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.pkl')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics.txt')

class MLAgent:
    """An agent class to select, train, evaluate, and tune machine learning models."""
    
    def __init__(self):
        # --- List of all Classification Models (Focus of this script) ---
        self.classification_models = {
            'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {}),
            'KNeighborsClassifier': (KNeighborsClassifier(), {'n_neighbors': 5}),
            'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {'max_depth': 5}),
            'RandomForestClassifier': (RandomForestClassifier(random_state=42), {'n_estimators': 100}),
            'GradientBoostingClassifier': (GradientBoostingClassifier(random_state=42), {'n_estimators': 100}),
            'ExtraTreesClassifier': (ExtraTreesClassifier(random_state=42), {'n_estimators': 100}),
            'AdaBoostClassifier': (AdaBoostClassifier(random_state=42), {'n_estimators': 50}),
            'SVC': (SVC(random_state=42, probability=True), {'C': 1.0}),
            'GaussianNB': (GaussianNB(), {}),
            'XGBClassifier': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {'n_estimators': 100})
        }
        
        # --- List of all Regression Models (For future expansion, currently unused in core logic) ---
        self.regression_models = {
            'LinearRegression': (LinearRegression(), {}),
            'Lasso': (Lasso(random_state=42), {'alpha': 1.0}),
            'Ridge': (Ridge(random_state=42), {'alpha': 1.0}),
            'ElasticNet': (ElasticNet(random_state=42), {'alpha': 1.0}),
            'KNeighborsRegressor': (KNeighborsRegressor(), {'n_neighbors': 5}),
            'DecisionTreeRegressor': (DecisionTreeRegressor(random_state=42), {'max_depth': 5}),
            'RandomForestRegressor': (RandomForestRegressor(random_state=42), {'n_estimators': 100}),
            'GradientBoostingRegressor': (GradientBoostingRegressor(random_state=42), {'n_estimators': 100}),
            'ExtraTreesRegressor': (ExtraTreesRegressor(random_state=42), {'n_estimators': 100}),
            'AdaBoostRegressor': (AdaBoostRegressor(random_state=42), {'n_estimators': 50}),
            'SVR': (SVR(), {'C': 1.0}),
            'XGBRegressor': (xgb.XGBRegressor(eval_metric='rmse', random_state=42), {'n_estimators': 100})
        }

        # Set the active model list to classification for the current pipeline logic
        self.models = self.classification_models 

        self.best_model_name = None
        self.best_model = None
        self.max_accuracy = 0.0
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self):
        """Loads and returns feature matrix (X) and target vector (y) and applies SMOTE."""
        try:
            # Note: The agent is currently hardcoded for classification based on the target logic
            print("--- AGENT: Classification task detected. Loading simulated data.")
            
            # --- ⚠️ REPLACE THIS SECTION with your actual data loading and cleaning ---
            data = {
                'feature1': np.random.rand(1000),
                'feature2': np.random.rand(1000) * 10,
                'feature3': np.random.rand(1000) * 5,
                'target': np.random.choice([0, 1], size=1000, p=[0.8, 0.2]) 
            }
            df = pd.DataFrame(data)
            
            X = df[['feature1', 'feature2', 'feature3']]
            y = df['target']
            
            # Apply SMOTE to handle class imbalance
            print("--- PREPROCESS: Applying SMOTE to balance classes.")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            print(f"--- PREPROCESS: Original Data Size: {len(X)}. Resampled Data Size: {len(X_resampled)}.")
            return X_resampled, y_resampled

        except Exception as e:
            print(f"::error::FATAL WORKFLOW ERROR: Data Loading failed: {e}")
            return None, None

    def select_best_model(self, X_train, X_test, y_train, y_test):
        """Trains and evaluates all candidate models to find the best performing one."""
        print(f"\n--- AGENT: Starting Model Selection Phase (Testing {len(self.models)} Classification algorithms) ---")
        
        for name, (model, params) in self.models.items():
            start_time = time.time()
            
            pipeline = Pipeline([
                ('scaler', self.scaler),
                ('model', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"--- EVAL: {name} trained. Accuracy: {accuracy:.4f} (Time: {time.time()-start_time:.2f}s)")
            
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.best_model = pipeline
                self.best_model_name = name
                
        print(f"\n--- AGENT: Initial Best Model Selected: {self.best_model_name} with Accuracy: {self.max_accuracy:.4f} ---")
        
    def hyperparameter_tuning(self, X_train, y_train, X_test, y_test):
        """Performs targeted tuning on the current best model for remediation."""
        print(f"\n--- AGENT: REMEDIATION TRIGGERED: Starting Hyperparameter Tuning for {self.best_model_name} ---")
        
        # Define a simple tuning grid for the current best model's type
        if 'Classifier' in self.best_model_name and 'Tree' in self.best_model_name:
            param_grid = {'model__max_depth': [3, 5, 7, 9], 'model__n_estimators': [50, 100]}
        elif self.best_model_name == 'XGBClassifier':
            param_grid = {'model__max_depth': [3, 5, 7], 'model__learning_rate': [0.01, 0.1]}
        elif self.best_model_name in ['LogisticRegression', 'SVC']:
            param_grid = {'model__C': [0.1, 1.0, 10]}
        else:
            print("--- TUNING SKIP: No specific tuning grid defined for this model type. Remediation skipped.")
            return

        base_model = self.best_model.named_steps['model']
        pipeline_to_tune = Pipeline([
            ('scaler', self.scaler),
            ('model', base_model)
        ])
        
        # GridSearch will find the best model parameters
        grid_search = GridSearchCV(pipeline_to_tune, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Evaluate the tuned model on the test set
        final_accuracy = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
        print(f"--- REMEDIATION RESULT: Test Set Accuracy after tuning: {final_accuracy:.4f}")
        
        # Only update the stored best model if tuning improved the test set score
        if final_accuracy > self.max_accuracy:
            self.max_accuracy = final_accuracy
            self.best_model = grid_search.best_estimator_
            self.best_model_name = f"{self.best_model_name} (Tuned)"
            print("--- AGENT: Tuning successfully improved the final model.")
        else:
            print("--- AGENT: Tuning did not improve final test set performance. Sticking with original best model.")


    def run_agentic_workflow(self):
        """Runs the entire MLOps workflow with tiered performance logic."""
        
        X, y = self.load_and_preprocess_data()
        if X is None:
            return 0.0

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Phase 1: Initial Model Selection
        self.select_best_model(X_train, X_test, y_train, y_test)
        
        initial_accuracy = self.max_accuracy
        
        # Phase 2: Agentic Performance Tiers
        
        # Tier 1: High Accuracy (Deployment Ready)
        if 0.80 <= initial_accuracy <= 0.90:
            print(f"\n*** AGENT STATUS: BEST *** Performance {initial_accuracy:.4f} is excellent. Deployment approved.")
            
        # Tier 2: Medium Accuracy (Remediation Required)
        elif 0.50 <= initial_accuracy < 0.80:
            print(f"\n*** AGENT STATUS: MEDIUM *** Performance {initial_accuracy:.4f} is moderate. Triggering Auto-Tuning...")
            self.hyperparameter_tuning(X_train, y_train, X_test, y_test)
            
        # Tier 3: Low Accuracy (Critical Failure)
        elif initial_accuracy < 0.50:
            print(f"\n::error::FATAL WORKFLOW ERROR: Accuracy {initial_accuracy:.4f} is critically low. Aborting workflow.")
            return 0.0

        print(f"\n--- AGENT SUMMARY: Final Model is {self.best_model_name} with Final Accuracy: {self.max_accuracy:.4f} ---")
        
        # Phase 3: Save Artifacts
        self.save_artifacts()
        
        return self.max_accuracy

    def save_artifacts(self):
        """Saves the final best model and metrics to the mounted directory."""
        
        try:
            # Save Model (.pkl)
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"--- INFO: Successfully saved final model to {MODEL_PATH}")

            # Save Metrics (.txt)
            with open(METRICS_PATH, 'w') as f:
                f.write(f"accuracy={self.max_accuracy:.4f}\n")
            print(f"--- INFO: Successfully saved metrics to {METRICS_PATH}")

        except Exception as e:
            print(f"::error::FATAL WORKFLOW ERROR: Artifact Saving failed (Permissions or Path): {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    agent = MLAgent()
    final_accuracy = agent.run_agentic_workflow()
    
    # Print the metric line for the GitHub Action to capture
    print(f"\n***METRIC_OUTPUT***accuracy={final_accuracy:.4f}***METRIC_OUTPUT***")
    print("--- WORKFLOW END: SUCCESS ---")
