import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import glob
import os

def load_data(file_pattern):
    files = glob.glob(file_pattern)
    data_list = []
    labels_list = []
    filenames = []
    
    for file in files:
        data = pd.read_csv(file)
        data_mean = data.mean().values

        filename = os.path.basename(file)
        label = filename.split('_')[-1].split('.')[0]
        labels_list.append(label)
        data_list.append(data_mean)
        filenames.append(filename)
    
    return np.array(data_list), labels_list, filenames

file_pattern = './workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/test/Bebop2_16g_1kdps_normalized_*.csv'

data, labels, filenames = load_data(file_pattern)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

labels_A = np.array([int(label[0]) for label in labels])
labels_B = np.array([int(label[1]) for label in labels])
labels_C = np.array([int(label[2]) for label in labels])
labels_D = np.array([int(label[3]) for label in labels])

def train_and_predict_kmeans(data, labels):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)
    predicted_labels = kmeans.predict(data)

    labels_mapping = {}
    for cluster in range(3):
        class_labels = [labels[i] for i in range(len(labels)) if predicted_labels[i] == cluster]
        if class_labels:
            majority_class = max(set(class_labels), key=class_labels.count)
            labels_mapping[cluster] = majority_class
        else:
            labels_mapping[cluster] = None

    adjusted_labels = [labels_mapping[cluster] if labels_mapping[cluster] is not None else -1 for cluster in predicted_labels]
    
    accuracy = np.mean([adjusted_labels[i] == labels[i] for i in range(len(labels)) if adjusted_labels[i] != -1])
    return accuracy, adjusted_labels

accuracy_A, predicted_labels_A = train_and_predict_kmeans(data_scaled[:, 0:6], labels_A)
print(f'Accuracy for propeller A: {accuracy_A:.2f}')

accuracy_B, predicted_labels_B = train_and_predict_kmeans(data_scaled[:, 6:12], labels_B)
print(f'Accuracy for propeller B: {accuracy_B:.2f}')

accuracy_C, predicted_labels_C = train_and_predict_kmeans(data_scaled[:, 12:18], labels_C)
print(f'Accuracy for propeller C: {accuracy_C:.2f}')

accuracy_D, predicted_labels_D = train_and_predict_kmeans(data_scaled[:, 18:24], labels_D)
print(f'Accuracy for propeller D: {accuracy_D:.2f}')

for i in range(len(filenames)):
    print(f"Nazwa pliku: {filenames[i]}")
    print(f"Rzeczywiste uszkodzenia: A={labels_A[i]}, B={labels_B[i]}, C={labels_C[i]}, D={labels_D[i]}")
    print(f"Przewidywane uszkodzenia: A={predicted_labels_A[i]}, B={predicted_labels_B[i]}, C={predicted_labels_C[i]}, D={predicted_labels_D[i]}")
    print("--------------")
