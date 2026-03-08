import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
import os

DATA_FILE = "d:/Desktop/25%/25%/bldc_ml_dataset.csv"
OUTPUT_DIR = "d:/Desktop/25%/25%"

def main():
    print("Loading data for visualization...")
    df = pd.read_csv(DATA_FILE).fillna(0)
    
    # Separate features and target
    drop_cols = ['sample_id', 'temp', 'label', 'rpm_mean', 'rpm_std']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y_class = df['label']
    y_temp = df['temp']
    
    X_train, X_test, y_class_train, y_class_test, y_temp_train, y_temp_test = train_test_split(
        X, y_class, y_temp, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Train models
    print("Training models to generate plot data...")
    temp_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    temp_model.fit(X_train, y_temp_train)
    y_temp_pred = temp_model.predict(X_test)
    
    class_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    class_model.fit(X_train, y_class_train)
    y_class_pred = class_model.predict(X_test)
    
    # Style settings
    sns.set_theme(style="whitegrid")
    
    # 1. Plot True vs Predicted Temperature (Regression)
    print("Generating Temperature Prediction Scatter Plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_temp_test, y_temp_pred, alpha=0.6, color='b')
    # Plot ideal line
    min_val = min(y_temp_test.min(), min(y_temp_pred))
    max_val = max(y_temp_test.max(), max(y_temp_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal Prediction (y=x)")
    plt.title("Sensorless Temperature Prediction vs True Temperature")
    plt.xlabel("True Internal Temperature (°C)")
    plt.ylabel("Predicted Temperature from Electrical Signals (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_temperature_prediction.png"), dpi=300)
    plt.close()
    
    # 2. Confusion Matrix for Fault Classification
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_class_test, y_class_pred, labels=["Healthy", "Degrading", "Short_Circuit"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Healthy", "Degrading", "Short Circuit"],
                yticklabels=["Healthy", "Degrading", "Short Circuit"])
    plt.title("Fault Classification Confusion Matrix")
    plt.ylabel("Actual State")
    plt.xlabel("Predicted State by Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 3. Feature Importances Bar Chart
    print("Generating Feature Importances Plot...")
    importances = class_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:10] # top 10
    top_features = X.columns[sorted_idx]
    top_importances = importances[sorted_idx]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title("Top 10 Most Important Features for Fault Detection")
    plt.xlabel("Random Forest Importance Score")
    plt.ylabel("Extracted Signal Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_feature_importances.png"), dpi=300)
    plt.close()
    
    # 4. Feature Drift Scatter (V(ec)_rms vs Temp)
    # Pick the most important feature to show how it drifts with temperature
    print("Generating Feature Drift Plot...")
    top_feature = top_features[0]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='temp', y=top_feature, hue='label', 
                    palette={"Healthy": "green", "Degrading": "orange", "Short_Circuit": "red"}, alpha=0.7)
    plt.title(f"How {top_feature} Drifts as Motor Heats Up")
    plt.xlabel("Simulated Temperature (°C)")
    plt.ylabel(f"{top_feature} Magnitude")
    plt.axvline(105, color='gray', linestyle='--', label='Degradation Start')
    plt.axvline(125, color='black', linestyle='--', label='Short Circuit Start')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_feature_drift.png"), dpi=300)
    plt.close()
    
    print("All visualizations generated successfully!")
    print("Check the folder for .png files.")

if __name__ == '__main__':
    main()
