import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import argparse
import glob
import json

import torch


# Parser argumentów
parser = argparse.ArgumentParser()
parser.add_argument('--propeller', type=str, required=True, help='Model to test')
args = parser.parse_args()

# Ścieżka zapisu modelu
save_dir = os.path.join('./models/', args.propeller)
os.makedirs(save_dir, exist_ok=True)
print(f"Save directory: {save_dir}")

file_pattern = f'/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train{args.propeller}/Bebop2_16g_1kdps_normalized_*.csv'
cols = [f'{args.propeller}_aX', f'{args.propeller}_aY', f'{args.propeller}_aZ', 
        f'{args.propeller}_gX', f'{args.propeller}_gY', f'{args.propeller}_gZ']

# load data
def load_data(data_pattern, columns):
    files = glob.glob(data_pattern)
    data_list = []
    print('Training on:')
    for file in files:
        print(file)
        data = pd.read_csv(file)
        propeller_data = data[columns]
        data_list.append(propeller_data)
    return pd.concat(data_list, axis=0)

df_train = load_data(file_pattern, cols)
print(df_train.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_train)

# Trenowanie modelu Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(scaled_data)

# Obliczenie błędów rekonstrukcji
decision_scores = -iso_forest.decision_function(scaled_data)
threshold = np.percentile(decision_scores, 95)

model_path = os.path.join(save_dir, 'isolation_forest_model.pth')
scaler_path = os.path.join(save_dir, 'scaler_params.npy')
threshold_path = os.path.join(save_dir, 'threshold.json')

# Po trenowaniu Isolation Forest
model_data = {
    'estimator_params': iso_forest.get_params(), 
    'estimator': iso_forest,
}

torch.save(model_data, './models/A/isolation_forest_model.pth')
print("Model saved to isolation_forest_model.pth")

np.save(scaler_path, [scaler.mean_, scaler.scale_])

# Zapisanie progu
with open(threshold_path, 'w') as f:
    json.dump({'threshold': float(threshold)}, f)

print(f"Model saved to {model_path}")
print(f"Scaler parameters saved to {scaler_path}")
print(f"Threshold saved to {threshold_path}")
print(f"Threshold value: {threshold}")
