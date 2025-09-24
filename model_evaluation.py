# src/model_evaluation.py
import pandas as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from utils import get_mood_to_genre_mapping, DEFAULT_DATASET_PATH
from data_cleaning import clean_and_preprocess_data # For consistent data
from sklearn.model_selection import train_test_split # To get test set

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True) # Ensure plot directory exists

def evaluate_model(model_path, X_test, y_test, label_encoder, plot_dir=PLOT_DIR):
    """
    Loads a trained model and evaluates it on the test set.
    Displays metrics and visualizations.
    """
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Making predictions on test set...")
    y_pred = model.predict(X_test)

    # --- Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")

    # Get original mood names for report
    target_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Confusion matrix plot saved to {plot_path}")

    # --- Feature Importance (if applicable) ---
    if hasattr(model, 'feature_importances_'):
        # Load feature names saved during preprocessing
        try:
            feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
            feature_names = joblib.load(feature_names_path)
        except:
             print("Warning: Could not load feature names for importance plot.")
             feature_names = [f"Feature_{i}" for i in range(len(model.feature_importances_))]

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, 'feature_importance.png')
        plt.savefig(plot_path)
        plt.show()
        print(f"Feature importance plot saved to {plot_path}")
    else:
        print("Model does not have feature importances attribute.")


if __name__ == "__main__":
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    # Need X_test and y_test. Re-run preprocessing/split for consistency.
    try:
        print("Re-processing data for evaluation...")
        X, y, df_processed, scaler, le, feature_names = clean_and_preprocess_data(filepath=DEFAULT_DATASET_PATH, save_scaler_encoder=False)
        # Re-split to get test set (use same random_state as training)
        X_train, X_test_eval, y_train, y_test_eval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        evaluate_model(model_path, X_test_eval, y_test_eval, le)

    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
