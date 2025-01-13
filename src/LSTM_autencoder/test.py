import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from lstm_autoencoder import LSTMAutoencoder
import torch.nn as nn
import json
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--propeller', type=str, default='A', help='Model to test')
args = parser.parse_args()

# Constants
SEQUENCE_LENGTH = 50
OVERLAP = 25

def create_sequences(data, seq_length, overlap):
    sequences = []
    for i in range(0, len(data) - seq_length + 1, seq_length - overlap):
        sequence = data[i:i + seq_length]
        sequences.append(sequence)
    return np.array(sequences)

# Load model and parameters
input_dim = 6
model = LSTMAutoencoder(input_size=input_dim)
model.load_state_dict(torch.load(f'./models/{args.propeller}/lstm_autoencoder_model.pth'))
model.eval()

scaler_params = np.load(f'./models/{args.propeller}/scaler_params.npy', allow_pickle=True)
scaler_mean, scaler_scale = scaler_params
scaler = lambda x: (x - scaler_mean) / scaler_scale

with open(f'models/{args.propeller}/threshold.json', 'r') as f:
    data = json.load(f)
    threshold = data['threshold']

# Test data paths
folder_path = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/test/'
skip_path = f'/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train{args.propeller}/'

# Metrics initialization
correct_detection = 0
incorrect_detection = 0
TP = TN = FP = FN = 0

cols = [f'{args.propeller}_aX', f'{args.propeller}_aY', f'{args.propeller}_aZ',
        f'{args.propeller}_gX', f'{args.propeller}_gY', f'{args.propeller}_gZ']

place_in_row = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[args.propeller]

# Test each file
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for csv_file in csv_files:
    if csv_file in os.listdir(skip_path):
        continue
        
    file_path = os.path.join(folder_path, csv_file)
    df_test = pd.read_csv(file_path)
    propeller_test = df_test[cols]
    
    # Preprocess and create sequences
    data_test = scaler(propeller_test.values)
    sequences = create_sequences(data_test, SEQUENCE_LENGTH, OVERLAP)
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    
    # Get reconstruction error
    with torch.no_grad():
        reconstruction = model(sequences_tensor)
        reconstruction_error = nn.MSELoss(reduction='none')(reconstruction, sequences_tensor)
        reconstruction_error = reconstruction_error.mean(dim=(1, 2)).numpy()
    
    # Detect anomalies
    anomalies = reconstruction_error > threshold
    damage = np.sum(anomalies)
    no_damage = len(anomalies) - damage
    
    # Get true label from filename
    filename_without_extension = csv_file.split('.')[0]
    parts = filename_without_extension.split('_')
    last_part = parts[-1]
    desired_character = last_part[place_in_row]
    
    # Classify and update metrics
    if damage > no_damage * 1.2:  # Using 20% threshold
        print(f"Damage detected in {csv_file}")
        if desired_character in ['1', '2']:
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

# Print metrics
accuracy = correct_detection / (correct_detection + incorrect_detection)
print(f'Correct detection: {correct_detection}')
print(f'Incorrect detection: {incorrect_detection}')
print(f'Accuracy: {accuracy}')
print(f'True Positives: {TP}')
print(f'True Negatives: {TN}')
print(f'False Positives: {FP}')
print(f'False Negatives: {FN}')

# Create confusion matrix
confusion_matrix = np.array([[TP, FN],
                           [FP, TN]])
labels = ['Uszkodzenie', 'Brak uszkodzenia']

fig, ax = plt.subplots()
cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
plt.colorbar(cax)

ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywista wartość')
plt.title(f'Accuracy: {accuracy:.3f}')

for (i, j), val in np.ndenumerate(confusion_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')

plt.savefig(f'./results/confusion_matrix{args.propeller}.png')