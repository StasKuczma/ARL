import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Ścieżka do folderu z danymi oraz plik treningowy
data_folder = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/'
training_file = 'Bebop2_16g_1kdps_normalized_0000.csv'  # Plik z danymi w normalnych warunkach

# Wczytanie danych treningowych
df_train = pd.read_csv(os.path.join(data_folder, training_file))

# Funkcja do trenowania Isolation Forest dla konkretnego śmigła
def train_isolation_forest(df_train, propeller_prefix):
    propeller_data = df_train[[f'{propeller_prefix}_aX', f'{propeller_prefix}_aY', f'{propeller_prefix}_aZ', 
                               f'{propeller_prefix}_gX', f'{propeller_prefix}_gY', f'{propeller_prefix}_gZ']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(propeller_data)

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(scaled_data)

    return iso_forest, scaler

# Funkcja do wykrywania anomalii dla danego śmigła w pliku testowym
def detect_anomalies_for_propeller(iso_forest, scaler, df, propeller_prefix):
    propeller_data = df[[f'{propeller_prefix}_aX', f'{propeller_prefix}_aY', f'{propeller_prefix}_aZ', 
                         f'{propeller_prefix}_gX', f'{propeller_prefix}_gY', f'{propeller_prefix}_gZ']]
    scaled_data = scaler.transform(propeller_data)

    anomalies = iso_forest.predict(scaled_data)
    damage_count = sum(anomalies == -1)
    no_damage_count = sum(anomalies == 1)

    if damage_count > no_damage_count:
        return "Damage detected"
    else:
        return "No damage detected"

# Lista plików do analizy, pomijając plik treningowy
data_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and f != training_file]

# Trenowanie modelu dla każdego śmigła
iso_forests = {}
scalers = {}
for propeller in ['A', 'B', 'C', 'D']:
    iso_forest, scaler = train_isolation_forest(df_train, propeller)
    iso_forests[propeller] = iso_forest
    scalers[propeller] = scaler

# Przejście przez każdy plik i wykrywanie anomalii
for file in data_files:
    print(f"\nProcessing file: {file}")
    df = pd.read_csv(os.path.join(data_folder, file))

    for propeller in ['A', 'B', 'C', 'D']:
        result = detect_anomalies_for_propeller(iso_forests[propeller], scalers[propeller], df, propeller)
        print(f"Propeller {propeller}: {result}")



