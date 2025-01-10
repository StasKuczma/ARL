import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--propeller', type=str, default='A', help='Model to test')
args = parser.parse_args()


# Wczytanie zapisanego modelu i parametrów
kmeans_centers = np.load('./models/'+args.propeller+'/kmeans_model.npy')
scaler_params = np.load('./models/'+args.propeller+'/scaler_params.npy', allow_pickle=True)
scaler_mean, scaler_scale = scaler_params
scaler = lambda x: (x - scaler_mean) / scaler_scale

with open('./models/'+args.propeller+'/threshold.json', 'r') as f:
    threshold = json.load(f)['threshold']


print('./models/'+args.propeller+'/kmeans_model.npy')
print('./models/'+args.propeller+'/scaler_params.npy')

# Funkcja obliczająca odległość do najbliższego centroidu
def compute_distances(data, centers):
    distances = []
    for point in data:
        centroid_distances = [np.linalg.norm(point - centroid) for centroid in centers]
        min_distance = min(centroid_distances)
        distances.append(min_distance)
    return np.array(distances)

# Inicjalizacja liczników
correct_detection = 0
incorrect_detection = 0
TP = 0
TN = 0
FP = 0
FN = 0

# Analiza plików testowych
test_folder = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/test/'
test_files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]

skip_path = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train'+args.propeller+'/'



place_in_row=0

place_in_row = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[args.propeller]

for test_file in test_files:

    if test_file in os.listdir(skip_path):
        continue
    file_path = os.path.join(test_folder, test_file)
    df_test = pd.read_csv(file_path)
    
    # Wybór danych dla śmigła
    cols = [args.propeller+'_aX', args.propeller+'_aY',args.propeller+'_aZ',args.propeller+'_gX',args.propeller+'_gY',args.propeller+'_gZ']
    propeller_test = df_test[cols]
    
    # Skalowanie danych testowych
    data_test = scaler(propeller_test.values)
    
    # Obliczenie odległości do centroidów
    distances = compute_distances(data_test, kmeans_centers)
    
    # Określenie anomalii
    anomalies = distances > threshold
    damage_count = np.sum(anomalies)

    normal_count = len(anomalies) - damage_count

    # Określenie rzeczywistego stanu na podstawie nazwy pliku
    true_state = test_file.split('_')[-1].split('.')[0][place_in_row]  
    
    # Klasyfikacja zdecydowana większość uszkodzona
    threshold_percentage = 0.25

    print(f'demage count: {damage_count}')
    print(f'in smaller: {(len(distances) * threshold_percentage)  }')
    is_damaged = damage_count > (len(distances) * threshold_percentage)   
    
    if is_damaged:
        print(f"Damage detected in {test_file}")
        if true_state in ['1', '2']:
            correct_detection += 1
            TP += 1
        else:
            incorrect_detection += 1
            FP += 1
    else:
        print(f"No damage detected in {test_file}")
        if true_state == '0':
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

labels = ['Damage', 'No Damage']

fig, ax = plt.subplots()
cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

plt.colorbar(cax)

ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)

plt.xlabel('Predykcja')
plt.ylabel('Rzetywista wartość')

plt.title('Accuracy:'+str(correct_detection / (correct_detection + incorrect_detection)))


for (i, j), val in np.ndenumerate(confusion_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')

plt.savefig('./results/confusion_matrix'+args.propeller+'.png')

plt.close()

