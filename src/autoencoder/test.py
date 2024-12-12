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
model.load_state_dict(torch.load('./models/C_propeller/autoencoder_model.pth'))
model.eval()

scaler_params = np.load('./models/C_propeller/scaler_params.npy', allow_pickle=True)
scaler_mean, scaler_scale = scaler_params
scaler = lambda x: (x - scaler_mean) / scaler_scale

with open('models/C_propeller/threshold.json', 'r') as f:
    data = json.load(f)
    threshold = data['threshold']

folder_path = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/test/'


csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

i=0

correct_detection = 0
incorrect_detection = 0

TP = 0
TN = 0
FP = 0
FN = 0

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    
    df_test = pd.read_csv(file_path)
    A_propeler_test = df_test[['C_aX', 'C_aY', 'C_aZ', 'C_gX', 'C_gY', 'C_gZ']]

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

    parts = filename_without_extension.split('_')

    last_part = parts[-1]

    desired_character = last_part[2]

    if damage > no_damage:
        print(f"Damage detected in {csv_file}")

        if desired_character == '1' or desired_character == '2':
            correct_detection += 1
            TP += 1
        else:
            incorrect_detection += 1
            FP += 1
    else:
        print(f"No damage detected in {csv_file}")
        if desired_character == '0':
            correct_detection += 1
            TN += 1
        else:
            incorrect_detection += 1
            FN += 1

print('Correct detection:', correct_detection)
print('Incorrect detection:', incorrect_detection)

print('Accuracy:', correct_detection / (correct_detection + incorrect_detection))

print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

confusion_matrix = np.array([[TP, FN],
                             [FP, TN]])

labels = ['Uszkodzenie', 'Brak uszkodzenia']

fig, ax = plt.subplots()
cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

plt.colorbar(cax)

ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)

plt.xlabel('Predykcja')
plt.ylabel('Rzetywista wartość')

for (i, j), val in np.ndenumerate(confusion_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')

plt.show()



