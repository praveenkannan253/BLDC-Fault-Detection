import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import json
import os

DATA_FILE = "d:/Desktop/BLDC/25%/bldc_ml_dataset.csv"
OUTPUT_METRICS_FILE = "d:/Desktop/BLDC/25%/evaluation_metrics.txt"

def evaluate_model():
    print(f"Loading dataset: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return
        
    df = pd.read_csv(DATA_FILE).fillna(0)
    
    # Separate features and target
    drop_cols = ['sample_id', 'temp', 'label', 'rpm_mean', 'rpm_std']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y_class = df['label']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Train the Fault Classification Model
    print("Training Fault Classification Model for Evaluation...")
    class_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    class_model.fit(X_train, y_class_train)
    
    # Predictions
    y_class_pred = class_model.predict(X_test)
    
    # Collect all metrics
    labels = sorted(list(y_class.unique())) # Typically: ['Degrading', 'Healthy', 'Short_Circuit']
    
    # 1. Accuracy
    acc = accuracy_score(y_class_test, y_class_pred)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_class_test, y_class_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    
    # Calculate True Positives, False Positives, False Negatives, True Negatives per class
    # Since it's multiclass, we do it one-vs-all for each class
    fp = cm.sum(axis=0) - np.diag(cm)  
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    
    # Compute Specificity and Sensitivity per class manually to be very explicit
    sensitivity = tp / (tp + fn) # Same as recall
    specificity = tn / (tn + fp)
    
    # 3. Precision, Recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(y_class_test, y_class_pred, labels=labels)
    
    # 4. Write everything to a comprehensive text block
    with open(OUTPUT_METRICS_FILE, 'w') as f:
        f.write("="*60 + "\n")
        f.write(" COMPREHENSIVE ML EVALUATION METRICS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Samples Tested: {len(X_test)}\n")
        f.write(f"Overall Accuracy:     {acc * 100:.2f}%\n\n")
        
        f.write("-"*60 + "\n")
        f.write(" 1. CONFUSION MATRIX\n")
        f.write("-"*60 + "\n")
        f.write(cm_df.to_string())
        f.write("\n\n")
        
        f.write("-"*60 + "\n")
        f.write(" 2. DETAILED METRICS BY CLASS\n")
        f.write("-"*60 + "\n")
        
        for i, label in enumerate(labels):
            f.write(f"Class: {label}\n")
            f.write(f"  - Support (True count): {support[i]}\n")
            f.write(f"  - True Positives (TP):  {tp[i]}\n")
            f.write(f"  - True Negatives (TN):  {tn[i]}\n")
            f.write(f"  - False Positives (FP): {fp[i]}\n")
            f.write(f"  - False Negatives (FN): {fn[i]}\n")
            f.write(f"  - Precision (PPV):      {precision[i]:.4f}\n")
            f.write(f"  - Recall / Sensitivity: {recall[i]:.4f}\n")
            f.write(f"  - Specificity (TNR):    {specificity[i]:.4f}\n")
            f.write(f"  - F1-Score:             {f1[i]:.4f}\n\n")
            
        f.write("-"*60 + "\n")
        f.write(" 3. STANDARD CLASSIFICATION REPORT OVERVIEW\n")
        f.write("-"*60 + "\n")
        f.write(classification_report(y_class_test, y_class_pred, labels=labels))
        
    print(f"Evaluation metrics successfully calculated and saved to: {OUTPUT_METRICS_FILE}")

if __name__ == "__main__":
    evaluate_model()
