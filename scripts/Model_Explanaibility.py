import pickle
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load the model
model_path = 'C:/Users/user/Desktop/Github/Adey_FraudDetection/mlruns/937789298497431940/6f3ae6530918427ca6449df2ed22ea83/artifacts/Random Forest/model.pkl'
with open(model_path, 'rb') as file:
    rf_model = pickle.load(file)

# Load the data for explanations
creditcard_df = pd.read_csv('C:/Users/user/Desktop/Github/Adey_FraudDetection/data/creditcard.csv')
X = creditcard_df.drop('Class', axis=1)  # Feature set
y = creditcard_df['Class']  # Target variable

# Check the features used for training the model
expected_feature_count = rf_model.n_features_in_ if hasattr(rf_model, 'n_features_in_') else X.shape[1]
print(f"Random Forest model expects {expected_feature_count} features.")

# Subsample the data for faster computation (optional)
sample_size = min(1000, len(X))  # Use a maximum of 1000 samples or less
X_sample = X.sample(n=sample_size, random_state=42)

# Adjust X_sample to match model's expected input features
if expected_feature_count < X_sample.shape[1]:
    print(f"Warning: Reducing input features from {X_sample.shape[1]} to {expected_feature_count}.")
    X_sample = X_sample.iloc[:, :expected_feature_count]  # Select only the first `expected_feature_count` features

# Using SHAP for Explainability
# Initialize SHAP explainer
explainer = shap.Explainer(rf_model, X_sample)

# Calculate SHAP values
shap_values = explainer(X_sample)

# Print shapes for debugging
print("SHAP values shape:", shap_values.shape)
print("X_sample shape:", X_sample.shape)

# Summary Plot
try:
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig('shap_summary_plot.png')  # Save the summary plot
    plt.close()  # Close the plot to free up memory
except Exception as e:
    print(f"Error in summary plot: {e}")

# Force Plot for a single prediction (e.g., first instance in sample)
shap.initjs()  # Initialize JavaScript visualization in Jupyter
try:
    # Use the expected value and the first SHAP value
    shap.plots.force(explainer.expected_value, shap_values.values[0], X_sample.iloc[0], show=False)
    plt.savefig('shap_force_plot.png')  # Save the force plot
    plt.close()  # Close the plot
except Exception as e:
    print(f"Error in force plot: {e}")

# Dependence Plot for a specific feature, e.g., the first feature
try:
    shap.dependence_plot(0, shap_values.values, X_sample, show=False)  # Adjusted to use the index
    plt.savefig('shap_dependence_plot.png')  # Save the dependence plot
    plt.close()  # Close the plot
except Exception as e:
    print(f"Error in dependence plot: {e}")

# Using LIME for Explainability
# Initialize LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X.iloc[:,:expected_feature_count].values,
    feature_names=X.columns[:expected_feature_count],
    class_names=['Not Fraud', 'Fraud'],
    mode='classification'
)

# Select an instance for LIME
instance = X_sample.iloc[0].values[:expected_feature_count]  # Ensure we take the correct number of features

# Explain the prediction for the selected instance
try:
    exp = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=rf_model.predict_proba
    )

    # Feature Importance Plot
    fig = exp.as_pyplot_figure()
    plt.savefig('lime_feature_importance_plot.png')  # Save the LIME plot
    plt.close()  # Close the plot
except Exception as e:
    print(f"Error in LIME explanation: {e}")

print("SHAP and LIME plots have been saved.")
