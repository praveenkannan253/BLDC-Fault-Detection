import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.fft import fft, fftfreq

# Configuration
DATA_DIR = "d:/Desktop/BLDC/25%"
OUTPUT_CSV = "d:/Desktop/BLDC/25%/bldc_ml_dataset.csv"
TS_STEP = 10e-6  # 10 us uniform sampling rate
START_TIME = 0.02  # seconds
END_TIME = 0.215  # seconds
WINDOW_SIZE = 0.005  # 5 ms (contains ~2 electrical cycles)
WINDOW_STEP = 0.001  # 1 ms step to vastly increase dataset size
FS = 1.0 / TS_STEP
FUNDAMENTAL_FREQ = 430.0  # Hz
FUND_BAND_HALF = 100.0  # Hz band around fundamental to look for peak

def calculate_thd(signal, fs):
    """Calculate Total Harmonic Distortion (THD) using FFT."""
    # Remove DC offset
    signal = signal - np.mean(signal)
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)[:N//2]
    amplitudes = 2.0/N * np.abs(yf[0:N//2])
    
    # Restrict to frequencies > 0
    amplitudes = amplitudes[1:]
    xf = xf[1:]
    
    # Find fundamental
    fund_idx_min = np.argmin(np.abs(xf - (FUNDAMENTAL_FREQ - FUND_BAND_HALF)))
    fund_idx_max = np.argmin(np.abs(xf - (FUNDAMENTAL_FREQ + FUND_BAND_HALF)))
    
    if fund_idx_min >= fund_idx_max: fund_idx_max = fund_idx_min + 1
    
    fund_idx = fund_idx_min + np.argmax(amplitudes[fund_idx_min:fund_idx_max])
    v1 = amplitudes[fund_idx]
    
    if v1 == 0:
        return 0.0
        
    # Total energy (sum of squares of all amplitudes)
    # Parseval's theorem: sum(amplitudes^2)
    # THD = sqrt( sum(V_harm^2) ) / V1
    # since sum(V^2) = V1^2 + sum(V_harm^2)
    total_energy = np.sum(amplitudes**2)
    harmonics_energy = total_energy - v1**2
    if harmonics_energy < 0:
        harmonics_energy = 0
        
    thd = np.sqrt(harmonics_energy) / v1
    return thd

def extract_features(window_df):
    features = {}
    
    current_cols = [c for c in window_df.columns if c.startswith('Ix')]
    bemf_cols = [c for c in window_df.columns if re.match(r'V\(e[abc]\)', c)]
    speed_col = 'V(wrpm)' if 'V(wrpm)' in window_df.columns else None
    
    # Current features
    for col in current_cols:
        sig = window_df[col].values
        rms = np.sqrt(np.mean(sig**2))
        features[f"{col}_rms"] = rms
        features[f"{col}_pk2pk"] = np.max(sig) - np.min(sig)
        features[f"{col}_std"] = np.std(sig)
        features[f"{col}_kurtosis"] = kurtosis(sig)
        features[f"{col}_crest"] = np.max(np.abs(sig)) / rms if rms > 0 else 0
        
    # Back EMF features
    for col in bemf_cols:
        sig = window_df[col].values
        features[f"{col}_rms"] = np.sqrt(np.mean(sig**2))
        features[f"{col}_pk2pk"] = np.max(sig) - np.min(sig)
        features[f"{col}_std"] = np.std(sig)
        features[f"{col}_thd"] = calculate_thd(sig, FS)
        
    # Speed features
    if speed_col:
        sig = window_df[speed_col].values
        features["rpm_mean"] = np.mean(sig)
        features["rpm_std"] = np.std(sig)
        
    return features

def extract_temp_from_filename(filename):
    basename = os.path.basename(filename)
    match = re.search(r'@(\d+)_', basename)
    if match:
        return int(match.group(1))
    return None

def process_file(filepath):
    print(f"Processing {filepath}...")
    # Read the file
    df = pd.read_csv(filepath, sep='\t')
    
    time = df['time'].values
    
    # Resample to uniform time grid
    uniform_time = np.arange(time[0], time[-1], TS_STEP)
    uniform_df = pd.DataFrame({'time': uniform_time})
    
    for col in df.columns:
        if col != 'time':
            uniform_df[col] = np.interp(uniform_time, time, df[col].values)
            
    # Remove startup transient
    steady_df = uniform_df[(uniform_df['time'] >= START_TIME) & (uniform_df['time'] <= END_TIME)]
    steady_time = steady_df['time'].values
    
    if len(steady_time) == 0:
        print(f"Warning: No data in steady state region for {filepath}")
        return []
        
    # Segment and extract features
    all_features = []
    temp = extract_temp_from_filename(filepath)
    
    t_start = START_TIME
    while t_start + WINDOW_SIZE <= END_TIME:
        t_end = t_start + WINDOW_SIZE
        window = steady_df[(steady_df['time'] >= t_start) & (steady_df['time'] < t_end)]
        
        if len(window) > 0:
            feats = extract_features(window)
            feats['temp'] = temp
            if temp in [25, 50, 60, 75, 90, 100]:
                feats['label'] = 'Healthy'
            elif temp in [110, 120]:
                feats['label'] = 'Degrading'
            elif temp is not None and temp >= 130:
                feats['label'] = 'Short_Circuit'
            else:
                feats['label'] = 'Unknown'
            feats['sample_id'] = f"{os.path.basename(filepath)}_{int(t_start*1000)}ms"
            all_features.append(feats)
            
        t_start += WINDOW_STEP
        
    return all_features

def main():
    files = glob.glob(os.path.join(DATA_DIR, "**", "BLDCM test@*.txt"), recursive=True)
    dataset = []
    
    for f in files:
        dataset.extend(process_file(f))
        
    if dataset:
        df_out = pd.DataFrame(dataset)
        # Reorder columns to put sample_id, temp, label at the end
        cols = list(df_out.columns)
        for c in ['sample_id', 'temp', 'label']:
            if c in cols:
                cols.remove(c)
                cols.insert(0, c) if c == 'sample_id' else cols.append(c)
        df_out = df_out[cols]
        
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully created dataset with {len(df_out)} samples.")
        print(f"Saved to: {OUTPUT_CSV}")
    else:
        print("No data extracted!")

if __name__ == '__main__':
    main()
