# src/utils.py
import pandas as pd
import os

# Define paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Gets the parent directory of src/
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, 'mood_music_dataset.csv')

def load_data(filepath=DEFAULT_DATASET_PATH):
    """Loads the dataset from a CSV file."""
    try:
        # Use on_bad_lines='skip' to handle potential formatting issues in the raw data
        df = pd.read_csv(filepath, on_bad_lines='skip')
        # Basic check for empty dataframe
        if df.empty:
            raise ValueError("The loaded DataFrame is empty.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: File at {filepath} is empty.")
        raise
    except Exception as e:
         print(f"An error occurred while loading the data: {e}")
         raise

def get_mood_to_genre_mapping():
    """Defines the mapping from mood to music genre."""
    return {
        'Happy': 'Pop',
        'Sad': 'Blues',
        'Stressed': 'Lo-fi',
        'Relaxed': 'Ambient'
    }


# Note for you what this code do :

# The src/utils.py file contains helper functions and definitions used by other scripts in the project. 
# Specifically, it defines the path to the dataset (DEFAULT_DATASET_PATH) 
# provides functions to load the data (load_data) 
# Also , it map moods to music genres (get_mood_to_genre_mapping).