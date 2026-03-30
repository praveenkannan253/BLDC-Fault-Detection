import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_FILE = "d:/Desktop/BLDC/25%/bldc_ml_dataset.csv"

def main():
    print(f"Loading dataset...")
    df = pd.read_csv(DATA_FILE).fillna(0)
    
    drop_cols = ['sample_id', 'temp', 'label', 'rpm_mean', 'rpm_std']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Remove crest factor and kurtosis as we've done in training
    cols_to_drop = [col for col in X.columns if 'kurtosis' in col or 'crest' in col]
    X = X.drop(columns=cols_to_drop)
    
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. Alphas Factor vs Accuracy (Excluding 0.0)
    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
    alpha_accs = []
    
    print("Evaluating Alpha vs Accuracy...")
    for alpha in alphas:
        rf = RandomForestClassifier(n_estimators=100, ccp_alpha=alpha, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        alpha_accs.append(acc * 100)
        print(f"  Alpha: {alpha}, Accuracy: {acc*100:.2f}%")
        
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, alpha_accs, marker='o', linestyle='-', color='blue', linewidth=2)
    plt.xscale('log') # Log scale better visualizes small alpha shifts
    plt.title('Impact of Pruning Factor (Alpha) on Prediction Accuracy')
    plt.xlabel('ccp_alpha (Log Scale)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("d:/Desktop/BLDC/25%/plot_alpha_vs_accuracy.png")
    plt.close()
    
    # 2. Number of Estimators vs Accuracy
    estimators = [1, 5, 10, 25, 50, 100, 150]
    est_accs = []
    
    print("Evaluating Estimators vs Accuracy...")
    for n_est in estimators:
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        est_accs.append(acc * 100)
        print(f"  Estimators: {n_est}, Accuracy: {acc*100:.2f}%")
        
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, est_accs, marker='s', linestyle='--', color='green', linewidth=2)
    plt.title('Impact of Number of Classifiers (Trees) on accuracy')
    plt.xlabel('n_estimators (Number of Trees)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.5)
    plt.savefig("d:/Desktop/BLDC/25%/plot_estimators_vs_accuracy.png")
    plt.close()
    
    print("Graphs saved successfully.")

if __name__ == "__main__":
    main()
