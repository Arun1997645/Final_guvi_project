# src/data_cleaning.py
# The StandardScaler in scikit-learn is a preprocessing technique used to standardize features by removing the mean and scaling to unit variance. This process is also known as "z-score normalization".
# SimpleImputer is a tool in scikit-learn that replaces missing values (often represented as NaN) in a dataset with a meaningful statistical value or a constant.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder # Label encoder for label to numerical conversion
from sklearn.impute import SimpleImputer # Added missing import
import joblib # To save preprocessing objects
import os

# --- Use absolute import by importing the utils module directly ---
# Python will find it because src/ is in the path when you run the script from src/
import utils # <--- Changed this line to absolute import

# Define output directory for models/preprocessors
# Use utils.PROJECT_ROOT which is defined in utils.py
MODEL_DIR = os.path.join(utils.PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True) # Create directory if it doesn't exist

def clean_and_preprocess_data(filepath=utils.DEFAULT_DATASET_PATH, save_scaler_encoder=True, output_dir=MODEL_DIR):
    """
    Loads, cleans, and preprocesses the data.
    Returns the processed DataFrame and fitted preprocessing objects.
    Saves preprocessors if save_scaler_encoder is True.
    """
    print("Loading data...")
    df = utils.load_data(filepath) # <--- Use the function from utils module ( Check before .py file i.e., utils.py)

    print("Initial data shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # --- 1. Handle Missing Values ---
    # Identify numerical and categorical columns
    numerical_cols = ['Heart Rate', 'Skin Temperature', 'Blink Rate']
    categorical_cols = ['Time of Day']
    target_col = 'Mood'

    # Check for missing values
    print("\nMissing values before imputation:")
    print(df.isnull().sum())

    # Impute numerical columns with median
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Impute categorical columns with mode (most frequent)
    # Let's use dropna for simplicity and robustness for this example for 'Time of Day'.
    df = df.dropna(subset=categorical_cols) # Drop rows with missing 'Time of Day'

    print("\nMissing values after imputation/dropna:")
    print(df.isnull().sum())

    # --- 2. Handle Outliers (using IQR method for numerical features) ---
    print("\nHandling outliers...")
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    for col in numerical_cols:
        initial_shape = df.shape
        df = remove_outliers_iqr(df, col)
        print(f"Removed {initial_shape[0] - df.shape[0]} outliers from '{col}'")

    print("Data shape after outlier removal:", df.shape)

    # --- 3. Encode Categorical Variables ---
    print("\nEncoding categorical variables...")
    # One-hot encode 'Time of Day'
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

    # Label encode the target 'Mood'
    le_mood = LabelEncoder()
    df_encoded['Mood_encoded'] = le_mood.fit_transform(df_encoded[target_col])

    # --- 4. Separate Features and Target ---
    feature_cols = [col for col in df_encoded.columns if col not in [target_col, 'Mood_encoded', 'Score']] # Exclude target and Score
    X = df_encoded[feature_cols]
    y = df_encoded['Mood_encoded'] # Use encoded target

    print("Final feature columns:", feature_cols)
    print("Final data shape (X):", X.shape)
    print("Target distribution:")
    print(df_encoded[target_col].value_counts()) # Show original mood names

    # --- 5. Scale Numerical Features ---
    print("\nScaling numerical features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # --- 6. Save preprocessing objects for later use (e.g., in Streamlit app) ---
    if save_scaler_encoder:
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        le_path = os.path.join(output_dir, 'label_encoder.pkl')
        feature_names_path = os.path.join(output_dir, 'feature_names.pkl')

        joblib.dump(scaler, scaler_path)
        joblib.dump(le_mood, le_path)
        joblib.dump(feature_cols, feature_names_path)
        print(f"\nSaved scaler to {scaler_path}")
        print(f"Saved label encoder to {le_path}")
        print(f"Saved feature names to {feature_names_path}")

    print("\nData cleaning and preprocessing complete.")
    return X_scaled_df, y, df_encoded, scaler, le_mood, feature_cols


if __name__ == "__main__":
    # This block runs when the script is executed directly
    try:
        X, y, df_processed, scaler, le, feature_names = clean_and_preprocess_data()
        # Optionally save the cleaned/processed DataFrame
        cleaned_data_path = os.path.join(os.path.dirname(utils.DEFAULT_DATASET_PATH), 'cleaned_mood_music_dataset.csv')
        df_processed.to_csv(cleaned_data_path, index=False)
        print(f"\nCleaned dataset saved as '{cleaned_data_path}'")
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
