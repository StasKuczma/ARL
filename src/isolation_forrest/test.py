import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--propeller', type=str, default='A', help='Model to test')
args = parser.parse_args()

# paths
model_path = f'./models/{args.propeller}/isolation_forest_model.npy'
scaler_path = f'./models/{args.propeller}/scaler_params.npy'
threshold_path = f'./models/{args.propeller}/threshold.json'

folder_path = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/test/'
skip_path = f'/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train{args.propeller}/'

# model loading

model_data = torch.load('./models/A/isolation_forest_model.pth')
iso_forest = model_data['estimator']

scaler_params = np.load(scaler_path, allow_pickle=True)
scaler_mean, scaler_scale = scaler_params
scaler = lambda x: (x - scaler_mean) / scaler_scale

with open(threshold_path, 'r') as f:
    data = json.load(f)
    threshold = data['threshold']

# Przygotowanie testu
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
cols = [f'{args.propeller}_aX', f'{args.propeller}_aY', f'{args.propeller}_aZ', 
        f'{args.propeller}_gX', f'{args.propeller}_gY', f'{args.propeller}_gZ']

place_in_row = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[args.propeller]

correct_detection = 0
incorrect_detection = 0
TP = 0
TN = 0
FP = 0
FN = 0

# Przetwarzanie plików testowych
for csv_file in csv_files:
    if csv_file in os.listdir(skip_path):
        continue

    file_path = os.path.join(folder_path, csv_file)
    df_test = pd.read_csv(file_path)
    propeller_data = df_test[cols]

    # Skalowanie danych
    data_test = scaler(propeller_data.values)

    # Obliczanie błędów rekonstrukcji
    decision_scores = -iso_forest.decision_function(data_test)
    anomalies = decision_scores > threshold

    # Analiza wyników
    damage = np.sum(anomalies)
    no_damage = len(anomalies) - damage

    filename_without_extension = csv_file.split('.')[0]
    parts = filename_without_extension.split('_')
    last_part = parts[-1]
    desired_character = last_part[place_in_row]

    print(f"Damage: {damage}, No damage: {no_damage}")

    if damage > no_damage * 2:  # Próg 20%
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

# Podsumowanie wyników
print('Correct detection:', correct_detection)
print('Incorrect detection:', incorrect_detection)
print('Accuracy:', correct_detection / (correct_detection + incorrect_detection))
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

# Macierz konfuzji
confusion_matrix = np.array([[TP, FN],
                             [FP, TN]])

labels = ['Uszkodzenie', 'Brak uszkodzenia']

fig, ax = plt.subplots()
cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
plt.colorbar(cax)

plt.xlabel('Predykcja')
plt.ylabel('Rzeczywista wartość')

plt.title('Accuracy:'+str(correct_detection / (correct_detection + incorrect_detection)))

for (i, j), val in np.ndenumerate(confusion_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')

plt.savefig(f'./results/confusion_matrix_{args.propeller}.png')
