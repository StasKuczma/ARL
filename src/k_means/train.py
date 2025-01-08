# train_kmeans.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import json

# Tworzenie katalogu na modele
save_dir = './models/C_propeller'
os.makedirs(save_dir, exist_ok=True)

# Wczytanie danych treningowych (2 przeloty)
df1 = pd.read_csv('/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train/Bebop2_16g_1kdps_normalized_0000.csv')
df2 = pd.read_csv('/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train/Bebop2_16g_1kdps_normalized_0122.csv')

# Wybór danych dla śmigła (przykład dla śmigła C)
cols = ['A_aX', 'A_aY', 'A_aZ', 'A_gX', 'A_gY', 'A_gZ']
propeller_data1 = df1[cols]
propeller_data2 = df2[cols]

# Połączenie danych treningowych
training_data = pd.concat([propeller_data1, propeller_data2], axis=0)

# Skalowanie danych
scaler = StandardScaler()
data_scaled = scaler.fit_transform(training_data)

# Trenowanie modelu K-means
n_clusters = 2  # Możesz dostosować liczbę klastrów
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data_scaled)

# Obliczanie odległości od centroidów
distances = []
for point in data_scaled:
    centroid_distances = [np.linalg.norm(point - centroid) for centroid in kmeans.cluster_centers_]
    min_distance = min(centroid_distances)
    distances.append(min_distance)

distances = np.array(distances)

threshold = distances.mean() + 1.5 * distances.std()

# Zapisywanie modelu i parametrów
np.save(os.path.join(save_dir, 'kmeans_model.npy'), kmeans.cluster_centers_)
np.save(os.path.join(save_dir, 'scaler_params.npy'), [scaler.mean_, scaler.scale_])

with open(os.path.join(save_dir, 'threshold.json'), 'w') as f:
    json.dump({'threshold': float(threshold)}, f)

print(f'Model saved. Threshold value: {threshold}')