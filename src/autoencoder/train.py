import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import Autoencoder
import os
import glob


from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--propeller', type=str, required=True, help='Model to test')
args = parser.parse_args()


save_dir = os.path.join('./models/',args.propeller)

print(save_dir)

os.makedirs(save_dir, exist_ok=True)


file_pattern = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train'+args.propeller+'/Bebop2_16g_1kdps_normalized_*.csv'
cols = [args.propeller+'_aX', args.propeller+'_aY',args.propeller+'_aZ',args.propeller+'_gX',args.propeller+'_gY',args.propeller+'_gZ']

def load_data(data, cols):
    
    files = glob.glob(data)
    data_list = []
    print('train on: ')
    for file in files:

        print(file)
        data = pd.read_csv(file)
        propeller_data = data[cols]

        data_list.append(propeller_data)

    return pd.concat(data_list, axis=0)

training_data=load_data(file_pattern, cols)

print(training_data.head())

scaler = StandardScaler()
data = scaler.fit_transform(training_data)

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

torch.save(model.state_dict(), './models/'+args.propeller+'/autoencoder_model.pth')

np.save('./models/'+args.propeller+'/scaler_params.npy', [scaler.mean_, scaler.scale_])

print('Model saved to autoencoder_model.pth')

with torch.no_grad():
    training_reconstruction = model(data_tensor)
    training_reconstruction_error = nn.MSELoss(reduction='none')(training_reconstruction, data_tensor).mean(dim=1).numpy()
threshold = np.percentile(training_reconstruction_error, 95)  


with open('./models/'+args.propeller+'/threshold.json', 'w') as f:
    json.dump({'threshold': threshold}, f)

print(threshold)


# plt.plot(loss_values)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()