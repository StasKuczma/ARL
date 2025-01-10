# train_kmeans.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import json
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--propeller', type=str, required=True, help='Model to test')
args = parser.parse_args()


save_dir = os.path.join('./models/',args.propeller)

print(save_dir)

os.makedirs(save_dir, exist_ok=True)

# Wczytanie danych treningowych (2 przeloty)
file_pattern = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train'+args.propeller+'/Bebop2_16g_1kdps_normalized_*.csv'
cols = [args.propeller+'_aX', args.propeller+'_aY',args.propeller+'_aZ',args.propeller+'_gX',args.propeller+'_gY',args.propeller+'_gZ']

def load_data(data, cols):
    
    files = glob.glob(data)
    data_list = []
    
    for file in files:
        data = pd.read_csv(file)
        propeller_data = data[cols]

        data_list.append(propeller_data)

    return pd.concat(data_list, axis=0)

training_data=load_data(file_pattern, cols)

print(training_data)

# Skalowanie danych
scaler = StandardScaler()
data_scaled = scaler.fit_transform(training_data)

# Trenowanie modelu K-means
n_clusters = 3  # Możesz dostosować liczbę klastrów
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data_scaled)

# Obliczanie odległości od centroidów
distances = []
for point in data_scaled:
    centroid_distances = [np.linalg.norm(point - centroid) for centroid in kmeans.cluster_centers_]
    min_distance = min(centroid_distances)
    distances.append(min_distance)

distances = np.array(distances)
# train_distances = calculate_distances(train_scaled, kmeans.cluster_centers_)

# threshold = distances.mean() + 1.5 * distances.std()
threshold = np.percentile(distances, 90)  

# Zapisywanie modelu i parametrów
np.save(os.path.join(save_dir, 'kmeans_model.npy'), kmeans.cluster_centers_)
np.save(os.path.join(save_dir, 'scaler_params.npy'), [scaler.mean_, scaler.scale_])

with open(os.path.join(save_dir, 'threshold.json'), 'w') as f:
    json.dump({'threshold': float(threshold)}, f)

print(f'Model saved. Threshold value: {threshold}')