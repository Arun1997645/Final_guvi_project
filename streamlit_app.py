# src/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from utils import get_mood_to_genre_mapping, DEFAULT_DATASET_PATH

# Define paths relative to the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# --- Load necessary objects ---
@st.cache_resource # Cache to avoid reloading on every interaction
def load_model_and_preprocessors(model_dir=MODEL_DIR):
    model_path = os.path.join(model_dir, 'best_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    le_path = os.path.join(model_dir, 'label_encoder.pkl')
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(le_path)
        feature_names = joblib.load(feature_names_path)
        return model, scaler, label_encoder, feature_names
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure the model training script has been run successfully.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model/preprocessors: {e}")
        st.stop()

def main():
    st.title("Mood-based Music Recommendation 🎵")
    st.write("Enter your physiological data to get a mood prediction and music recommendation.")

    # --- Suggestion: Add Information Section ---
    with st.expander("ℹ️ About the Moods", expanded=False):
        st.write("""
        This model was trained to recognize four moods based on physiological signals:
        *   **Happy:** Generally associated with moderate heart rate, moderate skin temperature, and moderate blink rate.
        *   **Sad:** Often characterized by lower heart rate, lower skin temperature, and lower blink rate.
        *   **Stressed:** Typically linked to higher heart rate, higher skin temperature, and higher blink rate.
        *   **Relaxed:** Usually indicated by lower heart rate, moderate skin temperature, and lower blink rate.
        *Note: These are general trends observed in the training data. Individual experiences may vary.*
        """)

    # Load model and preprocessors
    model, scaler, label_encoder, feature_names = load_model_and_preprocessors()

    # --- User Input ---
    st.header("Enter Physiological Data")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=0.0, max_value=200.0, value=70.0, step=1.0, help="Your current heart rate in beats per minute.")
    with col2:
        skin_temp = st.number_input("Skin Temperature (°C)", min_value=0.0, max_value=50.0, value=36.0, step=0.1, help="Your current skin temperature in degrees Celsius.")
    with col3:
        blink_rate = st.number_input("Blink Rate (blinks/min)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, help="Your approximate blink rate in blinks per minute.")
    with col4:
        time_of_day = st.selectbox("Time of Day", options=['Morning', 'Afternoon', 'Evening', 'Night'], help="Select the current time of day.")

    # --- Prediction and Recommendation ---
    if st.button("Predict Mood and Recommend Music"):
        # 1. Prepare input data
        input_data = pd.DataFrame({
            'Heart Rate': [heart_rate],
            'Skin Temperature': [skin_temp],
            'Blink Rate': [blink_rate],
            'Time of Day': [time_of_day]
        })

        # 2. Encode categorical variable (Time of Day)
        # Create a DataFrame with all possible one-hot columns, initialized to 0
        encoded_input = pd.get_dummies(input_data, columns=['Time of Day'], prefix='Time of Day')
        # Ensure all columns from training are present (in case a category is missing in this single input)
        for col in feature_names:
            if col not in encoded_input.columns:
                encoded_input[col] = 0
        # Reorder columns to match the training data
        encoded_input = encoded_input.reindex(columns=feature_names, fill_value=0)

        # 3. Scale the numerical features
        input_scaled = scaler.transform(encoded_input)

        # 4. Make prediction
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # 5. Decode the prediction
        predicted_mood = label_encoder.inverse_transform([prediction_encoded])[0]

        # 6. Get music recommendation
        mood_to_genre = get_mood_to_genre_mapping()
        recommended_genre = mood_to_genre.get(predicted_mood, "Unknown")

        # --- Display Results ---
        st.header("Results")
        st.subheader(f"Predicted Mood: {predicted_mood}")
        st.write(f"Recommended Music Genre: **{recommended_genre}**")

        # --- Suggestion: Add Confidence Indicator ---
        confidence = np.max(prediction_proba)
        st.progress(float(confidence)) # Convert to Python float
        st.write(f"Prediction Confidence: {confidence:.2%}")

        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame({
            'Mood': label_encoder.classes_,
            'Probability': prediction_proba
        }).sort_values(by='Probability', ascending=False)
        # Use index for bar chart
        st.bar_chart(prob_df.set_index('Mood')['Probability'])

if __name__ == "__main__":
    main()