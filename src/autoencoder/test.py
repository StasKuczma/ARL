import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import Autoencoder
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import json


input_dim = 6  # Input dimension used in training
model = Autoencoder(input_dim)
model.load_state_dict(torch.load('./autoencoder_model.pth'))
model.eval()

scaler_params = np.load('./scaler_params.npy', allow_pickle=True)
scaler_mean, scaler_scale = scaler_params
scaler = lambda x: (x - scaler_mean) / scaler_scale

df_test = pd.read_csv('/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/Bebop2_16g_1kdps_normalized_0022.csv')
A_propeler_test = df_test[['C_aX', 'C_aY', 'C_aZ', 'C_gX', 'C_gY', 'C_gZ']]
# A_propeler_test = df_test[['A_aX', 'A_aY', 'A_aZ', 'A_gX', 'A_gY', 'A_gZ']]

data_test = scaler(A_propeler_test.values)
data_test_tensor = torch.tensor(data_test, dtype=torch.float32)

# Perform inference
with torch.no_grad():
    reconstruction = model(data_test_tensor)
    reconstruction_error = nn.MSELoss(reduction='none')(reconstruction, data_test_tensor).mean(dim=1).numpy()

with open('threshold.json', 'r') as f:
    data = json.load(f)
    threshold = data['threshold']

# Detect anomalies
anomalies = reconstruction_error > threshold

damage = 0
no_damage = 0
for i, anomaly in enumerate(anomalies):
    if anomaly:
        damage += 1
    else:
        no_damage += 1

if damage > no_damage:
    print("Damage detected")
else:
    print("No damage detected")
