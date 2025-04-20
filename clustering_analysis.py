import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import io # For returning plot image
import base64 # For embedding plot in markdown

def run_clustering_analysis(csv_path):
    """Performs clustering analysis on the provided CSV and returns results including cluster descriptions."""
    print(f"Running Clustering Analysis on {csv_path}")
    try:
        df = pd.read_csv(csv_path)

        # Select numerical features for clustering
        features = ['satisfaction_level', 'last_evaluation', 'number_project',
                    'average_montly_hours', 'time_spend_company']

        # Check if needed columns exist
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            return {"summary": f"Error: Missing required columns in CSV for clustering: {missing_cols}", "clusters": [], "cluster_count": 0}

        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal k (let's fix it at 4 for simplicity, elbow/silhouette can be done offline or added back carefully)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10) # Set n_init explicitly
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        cluster_analysis = df.groupby('Cluster')[features].mean()
        cluster_sizes = df['Cluster'].value_counts().sort_index()

        # Generate textual descriptions for each cluster
        cluster_descriptions = []
        for i in range(optimal_k):
            desc = f"Cluster {i} (Size: {cluster_sizes.get(i, 0)}):\n"
            desc += cluster_analysis.loc[i].to_string() # Add mean values
            # Add qualitative interpretation (this part can be enhanced significantly)
            if cluster_analysis.loc[i, 'satisfaction_level'] > 0.7 and cluster_analysis.loc[i, 'last_evaluation'] > 0.7:
                desc += "\nInterpretation: Likely high performers, satisfied."
            elif cluster_analysis.loc[i, 'satisfaction_level'] < 0.5:
                desc += "\nInterpretation: Potential disengagement/low satisfaction."
            else:
                desc += "\nInterpretation: Mixed characteristics."
            cluster_descriptions.append(desc)

        # Optional: Generate and return plot as base64 string
        # fig, ax = plt.subplots(figsize=(8, 5))
        # scatter = ax.scatter(df['satisfaction_level'], df['last_evaluation'], c=df['Cluster'], cmap='viridis')
        # ax.set_xlabel('Satisfaction Level')
        # ax.set_ylabel('Last Evaluation')
        # ax.set_title('Employee Clusters')
        # legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        # ax.add_artist(legend1)
        # 
        # buf = io.BytesIO()
        # fig.savefig(buf, format='png')
        # plt.close(fig) # Close the figure to free memory
        # buf.seek(0)
        # image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        # plot_markdown = f'![Cluster Plot](data:image/png;base64,{image_base64})'

        # --- Compile Results --- 
        results = {
            "summary": f"Clustering analysis identified {optimal_k} distinct employee groups.",
            "clusters": cluster_descriptions,
            "cluster_count": optimal_k,
            # "plot_markdown": plot_markdown # Uncomment to include plot markdown
        }
        print("Clustering analysis function complete.")
        return results

    except Exception as e:
        print(f"Error during clustering analysis: {e}")
        return {"summary": f"Error during clustering analysis: {e}", "clusters": [], "cluster_count": 0}

# --- Remove or comment out script-level execution ---
# df = pd.read_csv('data/hr_analytics.csv')
# ... (rest of the script-level code removed) ...