import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_basic_dashboard(csv_path):
    """Loads data and generates a basic HR dashboard figure."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {csv_path}")
        return None
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    # Check for required columns
    required_cols = ['sales', 'satisfaction_level', 'left', 'last_evaluation', 'number_project', 
                     'average_montly_hours', 'time_spend_company', 'salary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Error: Missing required columns for basic dashboard: {missing_cols}")
        return None

    # Set up the main figure with subplots
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    try:
        # 1. Department-wise Distribution (Pie Chart)
        plt.subplot(2, 3, 1)
        dept_dist = df['sales'].value_counts()
        plt.pie(dept_dist, labels=dept_dist.index, autopct='%1.1f%%')
        plt.title('Department Distribution')

        # 2. Satisfaction Level Distribution (Histogram)
        plt.subplot(2, 3, 2)
        sns.histplot(data=df, x='satisfaction_level', hue='left', bins=30, kde=True)
        plt.title('Satisfaction Level Distribution')

        # 3. Correlation Heatmap
        plt.subplot(2, 3, 3)
        numerical_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 
                        'average_montly_hours', 'time_spend_company']
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')

        # 4. Salary Distribution by Department (Box Plot)
        plt.subplot(2, 3, 4)
        sns.boxplot(data=df, x='sales', y='average_montly_hours', hue='salary')
        plt.xticks(rotation=45)
        plt.title('Working Hours by Department and Salary Level')

        # 5. Satisfaction vs Evaluation (Scatter Plot)
        plt.subplot(2, 3, 5)
        sns.scatterplot(data=df, x='satisfaction_level', y='last_evaluation', 
                        hue='left', style='salary', alpha=0.6)
        plt.title('Satisfaction vs Evaluation')

        # 6. Project Count Distribution
        plt.subplot(2, 3, 6)
        sns.countplot(data=df, x='number_project', hue='left')
        plt.title('Project Count Distribution')

        plt.tight_layout()

    except Exception as e:
        plt.close(fig) # Close figure if error occurs during plotting
        st.error(f"Error generating basic dashboard plots: {e}")
        return None

    return fig

# --- Remove script-level execution ---
# df = pd.read_csv('data/hr_analytics.csv')
# ... (rest of the script-level code removed) ...