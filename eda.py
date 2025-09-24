
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import load_data, DEFAULT_DATASET_PATH

# Define output directory for plots
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True) # Create directory if it doesn't exist

def perform_eda(filepath=DEFAULT_DATASET_PATH, output_dir=PLOT_DIR):
    """
    Performs Exploratory Data Analysis on the dataset.
    Generates plots and prints summary statistics.
    Saves plots to the output_dir.
    """
    print("Loading data for EDA...")
    df = load_data(filepath)

    print("\n--- Dataset Overview ---")
    print(df.head())
    print(df.info())
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Mood Distribution ---")
    print(df['Mood'].value_counts())
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Mood', data=df)
    plt.title('Distribution of Moods')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'mood_distribution.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")


    # --- Numerical Feature Analysis ---
    numerical_cols = ['Heart Rate', 'Skin Temperature', 'Blink Rate']
    print("\n--- Numerical Feature Distributions ---")
    df[numerical_cols].hist(bins=20, figsize=(12, 8))
    plt.suptitle('Distribution of Numerical Features')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'numerical_distributions.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")

    print("\n--- Boxplots for Numerical Features ---")
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numerical_cols):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='Mood', y=col, data=df)
        plt.title(f'{col} by Mood')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'numerical_boxplots_by_mood.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")

    # --- Categorical Feature Analysis ---
    categorical_cols = ['Time of Day']
    print("\n--- Time of Day Distribution ---")
    print(df['Time of Day'].value_counts())
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Time of Day', data=df)
    plt.title('Distribution of Time of Day')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'time_of_day_distribution.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")

    print("\n--- Mood vs Time of Day Cross-tabulation ---")
    crosstab = pd.crosstab(df['Time of Day'], df['Mood'])
    print(crosstab)
    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
    plt.title('Mood vs Time of Day')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'mood_vs_time_heatmap.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")

    # --- Correlation Analysis (Numerical only) ---
    print("\n--- Correlation Matrix (Numerical Features) ---")
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")

    print("\nEDA complete. Plots saved.")


if __name__ == "__main__":
    try:
        perform_eda()
    except Exception as e:
        print(f"An error occurred during EDA: {e}")
