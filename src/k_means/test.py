# test_kmeans.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Wczytanie zapisanego modelu i parametrów
kmeans_centers = np.load('./models/A_propeller/kmeans_model.npy')
scaler_params = np.load('./models/A_propeller/scaler_params.npy', allow_pickle=True)
scaler_mean, scaler_scale = scaler_params
scaler = lambda x: (x - scaler_mean) / scaler_scale

with open('models/A_propeller/threshold.json', 'r') as f:
    threshold = json.load(f)['threshold']

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

for test_file in test_files:
    file_path = os.path.join(test_folder, test_file)
    df_test = pd.read_csv(file_path)
    
    # Wybór danych dla śmigła
    cols = ['A_aX', 'A_aY', 'A_aZ', 'A_gX', 'A_gY', 'A_gZ']
    propeller_test = df_test[cols]
    
    # Skalowanie danych testowych
    data_test = scaler(propeller_test.values)
    
    # Obliczenie odległości do centroidów
    distances = compute_distances(data_test, kmeans_centers)
    
    # Określenie anomalii
    anomalies = distances > threshold

    # print(f'Anomalie w pliku {test_file}: {np.sum(anomalies)}')
    # print(f'Odległości: {len(distances)}')
    damage_count = np.sum(anomalies)

    # print(f'Anomalie w pliku : {damage_count}')
    normal_count = len(anomalies) - damage_count
    # print(f'normal count: {normal_count}')

    # Określenie rzeczywistego stanu na podstawie nazwy pliku
    true_state = test_file.split('_')[-1].split('.')[0][0]  # Dla śmigła C
    
    # Klasyfikacja
    is_damaged = damage_count > normal_count
    
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

# Wyświetlenie wyników
print('\nWyniki klasyfikacji:')
print(f'Poprawne detekcje: {correct_detection}')
print(f'Niepoprawne detekcje: {incorrect_detection}')
print(f'Dokładność: {correct_detection / (correct_detection + incorrect_detection):.4f}')

# # Dodatkowe metryki
# precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# recall = TP / (TP + FN) if (TP + FN) > 0 else 0
# f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# print('\nDodatkowe metryki:')
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F1 Score: {f1_score:.4f}')

# # Wizualizacja macierzy pomyłek
# confusion_matrix = np.array([[TP, FN], [FP, TN]])
# labels = ['Uszkodzenie', 'Brak uszkodzenia']

# plt.figure(figsize=(8, 6))
# plt.matshow(confusion_matrix, cmap=plt.cm.Blues, fignum=1)
# plt.colorbar()

# plt.xticks(range(2), labels)
# plt.yticks(range(2), labels)
# plt.xlabel('Predykcja')
# plt.ylabel('Rzeczywista wartość')

# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, str(confusion_matrix[i, j]), 
#                 ha='center', va='center')

# plt.show()