import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import Autoencoder


from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/Bebop2_16g_1kdps_normalized_0000.csv')
scaler = StandardScaler()
data = scaler.fit_transform(df.values)

# Dane tylko z smigla A
A_propeler=df[['A_aX', 'A_aY', 'A_aZ', 'A_gX', 'A_gY', 'A_gZ']]
data = scaler.fit_transform(A_propeler.values)


data_tensor = torch.tensor(data, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = data.shape[1]
model = Autoencoder(input_dim)

# Funkcja straty i optymalizator
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Trening
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        # Dane wejściowe
        batch_data = batch[0]

        # Przewidywanie i obliczenie straty
        output = model(batch_data)
        loss = criterion(output, batch_data)

        # Optymalizacja
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')




# Testowanie modelu

df_test = pd.read_csv('/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/Bebop2_16g_1kdps_normalized_1100.csv')
A_propeler_test = df_test[['A_aX', 'A_aY', 'A_aZ', 'A_gX', 'A_gY', 'A_gZ']]

data_test = scaler.transform(A_propeler_test.values)  
data_test_tensor = torch.tensor(data_test, dtype=torch.float32)

model.eval() 
with torch.no_grad():
    reconstruction = model(data_test_tensor)
    reconstruction_error = nn.MSELoss(reduction='none')(reconstruction, data_test_tensor).mean(dim=1).numpy()

with torch.no_grad():
    training_reconstruction = model(data_tensor)
    training_reconstruction_error = nn.MSELoss(reduction='none')(training_reconstruction, data_tensor).mean(dim=1).numpy()
threshold = np.percentile(training_reconstruction_error, 95)  

# Wykrycie anomalii na danych testowych
anomalies = reconstruction_error > threshold

damage = 0
no_damage = 0

for i, anomaly in enumerate(anomalies):
    if anomaly:
        damage += 1
    else:
        no_damage += 1
if damage > no_damage:
    print("Damage detected")
else:
    print("No damage detected")
