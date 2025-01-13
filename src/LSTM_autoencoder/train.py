import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from lstm_autoencoder import LSTMAutoencoder
import os
import glob
import torch.nn as nn
import torch.optim as optim
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--propeller', type=str, required=True, help='Model to test')
args = parser.parse_args()

save_dir = os.path.join('./models/', args.propeller)
os.makedirs(save_dir, exist_ok=True)

# Data loading parameters
SEQUENCE_LENGTH = 50  # Number of time steps in each sequence
OVERLAP = 25  # Number of overlapping time steps between sequences

def load_data(file_pattern, cols):
    files = glob.glob(file_pattern)
    data_list = []
    print('train on: ')
    for file in files:
        print(file)
        data = pd.read_csv(file)
        propeller_data = data[cols]
        data_list.append(propeller_data)
    return pd.concat(data_list, axis=0)

def create_sequences(data, seq_length, overlap):
    sequences = []
    for i in range(0, len(data) - seq_length + 1, seq_length - overlap):
        sequence = data[i:i + seq_length]
        sequences.append(sequence)
    return np.array(sequences)

# Load and preprocess data
file_pattern = '/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/train'+args.propeller+'/Bebop2_16g_1kdps_normalized_*.csv'
cols = [args.propeller+'_aX', args.propeller+'_aY', args.propeller+'_aZ', 
        args.propeller+'_gX', args.propeller+'_gY', args.propeller+'_gZ']

training_data = load_data(file_pattern, cols)
print(training_data.head())

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(training_data)

# Create sequences
sequences = create_sequences(scaled_data, SEQUENCE_LENGTH, OVERLAP)
sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
dataset = TensorDataset(sequences_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = training_data.shape[1]
model = LSTMAutoencoder(input_size=input_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
num_epochs = 50
loss_values = []

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        batch_data = batch[0]
        
        # Forward pass
        output = model(batch_data)
        loss = criterion(output, batch_data)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    loss_values.append(avg_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')

# Save model and parameters
torch.save(model.state_dict(), os.path.join(save_dir, 'lstm_autoencoder_model.pth'))
np.save(os.path.join(save_dir, 'scaler_params.npy'), [scaler.mean_, scaler.scale_])

# Calculate reconstruction error threshold
with torch.no_grad():
    reconstruction = model(sequences_tensor)
    reconstruction_error = nn.MSELoss(reduction='none')(reconstruction, sequences_tensor)
    reconstruction_error = reconstruction_error.mean(dim=(1, 2)).numpy()
    threshold = np.percentile(reconstruction_error, 95)

with open(os.path.join(save_dir, 'threshold.json'), 'w') as f:
    json.dump({'threshold': float(threshold)}, f)

print(f'Threshold: {threshold}')