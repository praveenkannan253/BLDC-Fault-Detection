import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error

DATA_FILE = "d:/Desktop/BLDC/25%/bldc_ml_dataset.csv"

def main():
    print(f"Loading dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    # Fill any NaNs with 0
    df = df.fillna(0)
    
    # Features (Drop identifiers, RPM, and the targets)
    # The user wants to map temperature from Voltage and Current.
    drop_cols = ['sample_id', 'temp', 'label', 'rpm_mean', 'rpm_std']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    y_class = df['label']
    y_temp = df['temp']
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_temp_train, y_temp_test = train_test_split(
        X, y_class, y_temp, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # ---------------------------------------------------------
    # Model 1: Temperature Regression
    # ---------------------------------------------------------
    print("Training Temperature Regression Model...")
    temp_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    temp_model.fit(X_train, y_temp_train)
    
    y_temp_pred = temp_model.predict(X_test)
    mae = mean_absolute_error(y_temp_test, y_temp_pred)
    
    # ---------------------------------------------------------
    # Model 2: Fault Classification
    # ---------------------------------------------------------
    print("Training Fault Classification Model...")
    class_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    class_model.fit(X_train, y_class_train)
    
    y_class_pred = class_model.predict(X_test)
    acc = accuracy_score(y_class_test, y_class_pred)
    
    # Feature importances for classification
    importances = class_model.feature_importances_
    feature_names = X.columns
    sorted_idx = importances.argsort()[::-1]
    
    with open('train_results.txt', 'w') as f:
        f.write(f"Class Distribution:\n{y_class.value_counts().to_string()}\n")
        f.write(f"\nTested on {len(X_test)} samples (using ONLY Current and Voltage signals, NO RPM).\n")
        
        f.write("-" * 50 + "\n")
        f.write(" TEMPERATURE PREDICTION RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.2f} Degrees Celsius\n\n")
        f.write("Sample Predictions (True vs Predicted):\n")
        for true_t, pred_t in list(zip(y_temp_test, y_temp_pred))[:10]:
            f.write(f"  True: {true_t} C  ->  Predicted: {pred_t:.1f} C\n")
            
        f.write("\n" + "-" * 50 + "\n")
        f.write(" FAULT CLASSIFICATION RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Overall Model Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_class_test, y_class_pred))
        
        f.write("\nTop 10 Most Important Features for Fault Detection:\n")
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            f.write(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}\n")
            
    # Export metrics to a separate CSV file
    report_dict = classification_report(y_class_test, y_class_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("evaluation_metrics.csv", index=True)
            
    print("Training finished and results written to train_results.txt and evaluation_metrics.csv")

if __name__ == '__main__':
    main()
