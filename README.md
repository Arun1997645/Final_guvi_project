# Mood-based Music Recommendation using Physiological Signals

          This project classifies a user's mood (Happy, Sad, Stressed, Relaxed) using physiological signals such as Heart Rate, Skin Temperature, and Blink Rate. After predicting the mood, the system recommends a suitable music genre to enhance the user's emotional state.

## Business Use Cases

          - Smart Music Apps (e.g., Spotify integration)
          - Mental Health and Wellness Platforms
          - Wearable Devices (Smartwatches)
          - Productivity Tools (Office ambiance)

## Dataset

          The dataset `mood_music_dataset.csv` contains simulated physiological data and associated moods.

## Project Structure

          │
          ├── data/
          │   └── mood_music_dataset.csv  <-- Place your CSV file here
          │
          ├── src/
          │   ├── __init__.py
          │   ├── utils.py
          │   ├── data_cleaning.py
          │   ├── eda.py
          │   ├── model_training.py
          │   ├── model_evaluation.py
          │   └── streamlit_app.py
          │
          ├── models/ (This folder will be created automatically when you run training)
          │   ├── best_model.pkl
          │   ├── scaler.pkl
          │   ├── label_encoder.pkl
          │   └── feature_names.pkl
          │
          ├── notebooks/ (Optional, for exploratory work)
          │
          ├── requirements.txt
          ├── README.md
          └── .gitignore (Optional, but recommended)

## Setup Instructions

          1. Clone the repository:  
             `git clone <your-repo-url>` (if using Git)
          
          2. Create a virtual environment (recommended):  
             ```bash
             python -m venv mood_music_env
             ```
             Activate the environment:  
             - On macOS/Linux:  
               `source mood_music_env/bin/activate`
             - On Windows:  
               `mood_music_env\Scripts\activate`
          
          3. Install dependencies:  
             ```bash
             pip install -r requirements.txt
             ```
          
          4. Ensure `mood_music_dataset.csv` is in the project root directory.

## Running the Analysis

            - Data Cleaning:  
              `python data_cleaning.py`
            
            - EDA:  
              `python eda.py`  
              (Generates plots, might require manual viewing)
            
            - Model Training:  
              `python model_training.py`  
              (Saves the model)
            
            - Model Evaluation:  
              `python model_evaluation.py`  
              (Prints metrics, shows plots)

## Running the Streamlit App

            streamlit run streamlit_app.py
