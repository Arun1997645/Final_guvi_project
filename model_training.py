# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # To save the model
import os
from data_cleaning import clean_and_preprocess_data # Reuse cleaning logic
from utils import DEFAULT_DATASET_PATH

# Define output directory for models
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure directory exists

def train_models(X, y, output_dir=MODEL_DIR):
    """
    Trains multiple classification models and selects the best one based on accuracy.
    Saves the best model.
    """
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    best_model = None
    best_score = 0
    best_model_name = ""
    results = {}

    print("\n--- Training Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        results[name] = {'model': model, 'accuracy': acc, 'predictions': y_pred, 'y_test': y_test}

        if acc > best_score:
            best_score = acc
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with Accuracy: {best_score:.4f}")

    # Save the best model from initial training
    best_model_path = os.path.join(output_dir, 'best_model.pkl')
    joblib.dump(best_model, best_model_path)
    print(f"\nBest model ({best_model_name}) saved as '{best_model_path}'")

    return best_model, best_model_name, results, X_test, y_test


if __name__ == "__main__":
    try:
        # Load and preprocess data
        X, y, df_processed, scaler, le, feature_names = clean_and_preprocess_data(filepath=DEFAULT_DATASET_PATH, save_scaler_encoder=True)

        # Train models
        best_model, best_name, results, X_test_final, y_test_final = train_models(X, y)

        # Hyperparameter tuning could be added here if desired

    except Exception as e:
        print(f"An error occurred during model training: {e}")
