import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import io # To capture classification report string

def run_attrition_analysis(csv_path):
    """Performs attrition analysis on the provided CSV file and returns key results."""
    print(f"Running Attrition Analysis on {csv_path}")
    try:
        df = pd.read_csv(csv_path)

        # Calculate turnover rate
        turnover_rate = df['left'].mean() * 100
        turnover_finding = f'Employee Turnover Rate: {turnover_rate:.2f}%'

        # --- Feature Engineering & Model Training ---
        # Select features for prediction
        # Ensure these columns exist in the uploaded CSV
        base_features = ['satisfaction_level', 'last_evaluation', 'number_project',
                         'average_montly_hours', 'time_spend_company', 'Work_accident',
                         'promotion_last_5years']
        categorical_features = ['sales', 'salary'] # Assuming these exist

        # Check if needed columns exist
        required_cols = base_features + categorical_features + ['left']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"summary": f"Error: Missing required columns in CSV: {missing_cols}", "key_findings": [], "accuracy": None, "top_features": []}

        df_processed = pd.get_dummies(df, columns=categorical_features, dummy_na=False)

        # Prepare features (X) and target (y)
        feature_cols = base_features + [col for col in df_processed.columns if any(cat_feat in col for cat_feat in categorical_features)]
        X = df_processed[feature_cols]
        y = df_processed['left']

        # Split the data
        # Note: Consider stratification if 'left' is imbalanced
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialize and train the Random Forest model
        rf_model = RandomForestClassifier(max_depth= 25, max_features= 'log2', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 267, random_state=42)
        rf_model.fit(X_train, y_train)

        # --- Evaluation --- 
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_finding = f'Prediction Model Accuracy: {accuracy:.2f}'

        # Get classification report as string
        report_string = classification_report(y_test, y_pred)
        # Simple summary string (can be improved)
        report_summary = "Classification Report Summary:\n" + report_string 

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        top_features = feature_importance.head(5) # Get top 5
        top_features_finding = "Top 5 factors influencing attrition prediction:\n" + top_features.to_string(index=False)

        # --- Compile Results --- 
        results = {
            "summary": f"Attrition analysis complete. Turnover: {turnover_rate:.2f}%, Model Accuracy: {accuracy:.2f}",
            "key_findings": [
                turnover_finding,
                accuracy_finding,
                top_features_finding,
                # report_summary # Classification report can be quite long for a finding
            ],
            "accuracy": accuracy,
            "top_features": top_features.to_dict('records') # Return structured data if needed elsewhere
        }
        print("Attrition analysis function complete.")
        return results

    except Exception as e:
        print(f"Error during attrition analysis: {e}")
        return {"summary": f"Error during attrition analysis: {e}", "key_findings": [], "accuracy": None, "top_features": []}

# --- Remove or comment out script-level execution ---
# df = pd.read_csv('data/hr_analytics.csv')
# ... (rest of the script-level code removed) ...

