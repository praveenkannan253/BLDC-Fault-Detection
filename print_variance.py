import pandas as pd
import json

DATA_FILE = r"d:/Desktop/BLDC/25%/bldc_ml_dataset.csv"
OUT_FILE = r"d:/Desktop/BLDC/var_results.txt"

def main():
    df = pd.read_csv(DATA_FILE)
    
    # Drop non-feature columns
    drop_cols = ['sample_id', 'temp', 'label', 'rpm_mean', 'rpm_std']
    df_features = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Calculate variances
    variances = df_features.var()
    
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        for feat, var in variances.items():
            f.write(f"{feat:<20}: {var:.6f}\n")
        f.write("\nSummary:\n")
        f.write(variances.describe().to_string())

if __name__ == "__main__":
    main()
