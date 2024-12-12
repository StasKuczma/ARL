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
import matplotlib.pyplot as plt



df = pd.read_csv('/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train/Bebop2_16g_1kdps_normalized_0000.csv')
df2 = pd.read_csv('/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train/Bebop2_16g_1kdps_normalized_1102.csv')
scaler = StandardScaler()
data = scaler.fit_transform(df.values)

# Dane tylko z smigla A
A_propeler=df[['C_aX', 'C_aY', 'C_aZ', 'C_gX', 'C_gY', 'C_gZ']]
A2_propeler=df2[['C_aX', 'C_aY', 'C_aZ', 'C_gX', 'C_gY', 'C_gZ']]

A_propeler = pd.concat([A_propeler], axis=0)

data = scaler.fit_transform(A_propeler.values)

data_tensor = torch.tensor(data, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

input_dim = data.shape[1]
model = Autoencoder(input_dim)

# Funkcja straty i optymalizator
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_values = []

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
    loss_values.append(total_loss/len(dataloader))

torch.save(model.state_dict(), './models/C_propeller/autoencoder_model.pth')

np.save('./models/C_propeller/scaler_params.npy', [scaler.mean_, scaler.scale_])

print('Model saved to autoencoder_model.pth')

with torch.no_grad():
    training_reconstruction = model(data_tensor)
    training_reconstruction_error = nn.MSELoss(reduction='none')(training_reconstruction, data_tensor).mean(dim=1).numpy()
threshold = np.percentile(training_reconstruction_error, 85)  


with open('./models/C_propeller/threshold.json', 'w') as f:
    json.dump({'threshold': threshold}, f)

print(threshold)


plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()