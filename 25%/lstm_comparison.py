import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

DATA_FILE = "d:/Desktop/BLDC/25%/bldc_ml_dataset.csv"
OUTPUT_COMP_FILE = "d:/Desktop/BLDC/25%/model_comparison.txt"

def calculate_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp = cm.sum(axis=0) - np.diag(cm)  
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    
    # Handle div by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sensitivity = np.where((tp + fn) > 0, tp / (tp + fn), 0)
        specificity = np.where((tn + fp) > 0, tn / (tn + fp), 0)
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'acc': acc,
        'cm': cm,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'support': support
    }

def main():
    print(f"Loading dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE).fillna(0)
    
    drop_cols = ['sample_id', 'temp', 'label', 'rpm_mean', 'rpm_std']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y_class = df['label']
    
    # Needs to scale features for Neural Network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Label encode y for LSTM
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_class)
    labels = le.classes_ # Expected: ['Degrading', 'Healthy', 'Short_Circuit']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Evaluate RF
    rf_metrics = calculate_metrics(y_test, y_pred_rf, range(len(labels)))
    
    print("--- Training LSTM ---")
    # Reshape for LSTM: (samples, time_steps, features)
    # We treat each feature vector as 1 timestep of multiple features.
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Convert labels to categorical (one-hot) for TF
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(labels))
    
    # Build LSTM Model
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), activation='relu', return_sequences=True))
    lstm_model.add(LSTM(32, activation='relu'))
    lstm_model.add(Dense(16, activation='relu'))
    lstm_model.add(Dense(len(labels), activation='softmax'))
    
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train
    print("Training LSTM Neural Network...")
    lstm_model.fit(X_train_lstm, y_train_cat, epochs=100, batch_size=32, verbose=0, validation_split=0.1)
    
    # Predict
    y_pred_lstm_prob = lstm_model.predict(X_test_lstm)
    y_pred_lstm = np.argmax(y_pred_lstm_prob, axis=1)
    
    # Evaluate LSTM
    lstm_metrics = calculate_metrics(y_test, y_pred_lstm, range(len(labels)))
    
    print(f"Writing comparison to {OUTPUT_COMP_FILE}")
    with open(OUTPUT_COMP_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" EXPLORATORY MODEL COMPARISON: RANDOM FOREST vs LSTM Neural Network\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Samples Tested: {len(y_test)}\n\n")
        
        # Overall comparison
        f.write("-"*80 + "\n")
        f.write(" 1. OVERALL ACCURACY\n")
        f.write("-"*80 + "\n")
        f.write(f"Random Forest Accuracy: {rf_metrics['acc'] * 100:.2f}%\n")
        f.write(f"LSTM Accuracy:          {lstm_metrics['acc'] * 100:.2f}%\n\n")
        
        f.write("-"*80 + "\n")
        f.write(" 2. DETAILED METRICS BY CLASS (SIDE-BY-SIDE)\n")
        f.write("-"*80 + "\n")
        
        for i, label in enumerate(labels):
            f.write(f"=== CLASS: {label} ===\n")
            f.write(f"Support (True Count): {rf_metrics['support'][i]}\n\n")
            
            f.write(f"{'Metric':<25} | {'Random Forest':<15} | {'LSTM':<15}\n")
            f.write("-" * 65 + "\n")
            f.write(f"{'True Positives (TP)':<25} | {rf_metrics['tp'][i]:<15} | {lstm_metrics['tp'][i]:<15}\n")
            f.write(f"{'True Negatives (TN)':<25} | {rf_metrics['tn'][i]:<15} | {lstm_metrics['tn'][i]:<15}\n")
            f.write(f"{'False Positives (FP)':<25} | {rf_metrics['fp'][i]:<15} | {lstm_metrics['fp'][i]:<15}\n")
            f.write(f"{'False Negatives (FN)':<25} | {rf_metrics['fn'][i]:<15} | {lstm_metrics['fn'][i]:<15}\n")
            f.write(f"{'Precision (PPV)':<25} | {rf_metrics['precision'][i]:<15.4f} | {lstm_metrics['precision'][i]:<15.4f}\n")
            f.write(f"{'Recall / Sensitivity':<25} | {rf_metrics['recall'][i]:<15.4f} | {lstm_metrics['recall'][i]:<15.4f}\n")
            f.write(f"{'Specificity (TNR)':<25} | {rf_metrics['specificity'][i]:<15.4f} | {lstm_metrics['specificity'][i]:<15.4f}\n")
            f.write(f"{'F1-Score':<25} | {rf_metrics['f1'][i]:<15.4f} | {lstm_metrics['f1'][i]:<15.4f}\n")
            f.write("\n")
            
        f.write("-"*80 + "\n")
        f.write(" 3. CONFUSION MATRICES\n")
        f.write("-"*80 + "\n")
        
        f.write("--- Random Forest ---\n")
        rf_cm_df = pd.DataFrame(rf_metrics['cm'], index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
        f.write(rf_cm_df.to_string())
        f.write("\n\n")
        
        f.write("--- LSTM Neural Network ---\n")
        lstm_cm_df = pd.DataFrame(lstm_metrics['cm'], index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
        f.write(lstm_cm_df.to_string())
        f.write("\n")

if __name__ == "__main__":
    main()
