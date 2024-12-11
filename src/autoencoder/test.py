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
import os



input_dim = 6  
model = Autoencoder(input_dim)
model.load_state_dict(torch.load('./autoencoder_model.pth'))
model.eval()

scaler_params = np.load('./scaler_params.npy', allow_pickle=True)
scaler_mean, scaler_scale = scaler_params
scaler = lambda x: (x - scaler_mean) / scaler_scale

with open('threshold.json', 'r') as f:
    data = json.load(f)
    threshold = data['threshold']

folder_path = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/test/'


csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

i=0

correct_detection = 0
incorrect_detection = 0

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    
    df_test = pd.read_csv(file_path)
    A_propeler_test = df_test[['A_aX', 'A_aY', 'A_aZ', 'A_gX', 'A_gY', 'A_gZ']]

    data_test = scaler(A_propeler_test.values)
    data_test_tensor = torch.tensor(data_test, dtype=torch.float32)

    with torch.no_grad():
        reconstruction = model(data_test_tensor)
        reconstruction_error = nn.MSELoss(reduction='none')(reconstruction, data_test_tensor).mean(dim=1).numpy()
    
    anomalies = reconstruction_error > threshold

    damage = 0
    no_damage = 0
    for i, anomaly in enumerate(anomalies):
        if anomaly:
            damage += 1
        else:
            no_damage += 1

    filename_without_extension = csv_file.split('.')[0]

    # Split the filename by underscores
    parts = filename_without_extension.split('_')

    # Get the last part (which contains '2000')
    last_part = parts[-1]

    # Get the character at the desired position (index 1 for '2' in '2000')
    desired_character = last_part[0]

    if damage > no_damage:
        print(f"Damage detected in {csv_file}")

        if desired_character == '1' or desired_character == '2':
            correct_detection += 1
        else:
            incorrect_detection += 1
    else:
        print(f"No damage detected in {csv_file}")
        if desired_character == '0':
            correct_detection += 1
        else:
            incorrect_detection += 1

print('Correct detection:', correct_detection)
print('Incorrect detection:', incorrect_detection)

print('Accuracy:', correct_detection / (correct_detection + incorrect_detection))



