import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import gridspec

def generate_extended_dashboard(csv_path):
    """Loads data and generates an extended HR dashboard figure."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {csv_path}")
        return None
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    # Check for required columns
    required_cols = ['time_spend_company', 'left', 'sales', 'Work_accident', 
                     'promotion_last_5years', 'number_project', 'average_montly_hours', 
                     'satisfaction_level', 'last_evaluation', 'salary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Error: Missing required columns for extended dashboard: {missing_cols}")
        return None

    # Set up the main figure with subplots
    plt.style.use('default')
    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    gs.update(wspace=0.3, hspace=0.4)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    try:
        # 1. Time Spend in Company Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=df, x='time_spend_company', hue='left', bins=10, kde=True, ax=ax1)
        ax1.set_title('Employee Tenure Distribution')

        # 2. Work Accident Analysis by Department
        ax2 = fig.add_subplot(gs[0, 1])
        accident_by_dept = df.groupby('sales')['Work_accident'].mean().sort_values(ascending=False)
        ax2.bar(accident_by_dept.index, accident_by_dept.values)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_title('Work Accident Rate by Department')

        # 3. Promotion vs Retention
        ax3 = fig.add_subplot(gs[0, 2])
        promotion_retention = pd.crosstab(df['promotion_last_5years'], df['left'])
        promotion_retention.plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_title('Impact of Promotion on Retention')

        # 4. Average Monthly Hours by Project Count
        ax4 = fig.add_subplot(gs[1, 0])
        sns.boxplot(data=df, x='number_project', y='average_montly_hours', hue='left', ax=ax4)
        ax4.set_title('Working Hours vs Project Count')

        # 5. Satisfaction Level Over Time
        ax5 = fig.add_subplot(gs[1, 1])
        sns.boxplot(data=df, x='time_spend_company', y='satisfaction_level', hue='left', ax=ax5)
        ax5.set_title('Satisfaction Trends Over Time')

        # 6. Department Performance (Evaluation)
        ax6 = fig.add_subplot(gs[1, 2])
        sns.boxplot(data=df, x='sales', y='last_evaluation', hue='left', ax=ax6)
        ax6.tick_params(axis='x', rotation=45)
        ax6.set_title('Department Performance Distribution')

        # 7. Project Load Analysis
        ax7 = fig.add_subplot(gs[2, 0])
        project_dist = df.groupby(['sales', 'number_project']).size().unstack()
        project_dist.plot(kind='bar', stacked=True, ax=ax7)
        ax7.tick_params(axis='x', rotation=45)
        ax7.set_title('Project Distribution Across Departments')

        # 8. Salary Level Impact
        ax8 = fig.add_subplot(gs[2, 1])
        salary_impact = pd.crosstab([df['salary'], df['sales']], df['left'])
        salary_impact.plot(kind='bar', stacked=True, ax=ax8)
        ax8.tick_params(axis='x', rotation=45)
        ax8.set_title('Salary Level Impact on Retention')

        # 9. High Performers Analysis
        ax9 = fig.add_subplot(gs[2, 2])
        sns.scatterplot(data=df[df['last_evaluation'] > 0.8],
                        x='average_montly_hours',
                        y='satisfaction_level',
                        hue='left',
                        size='number_project',
                        alpha=0.6,
                        ax=ax9)
        ax9.set_title('High Performers Analysis')

    except Exception as e:
        plt.close(fig) # Close figure if error occurs during plotting
        st.error(f"Error generating extended dashboard plots: {e}")
        return None

    return fig

# --- Remove script-level execution ---
# df = pd.read_csv('data/hr_analytics.csv')
# ... (rest of the script-level code removed) ...