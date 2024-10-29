import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv('/workspace/UAV_measurement_data/Parrot_Bebop_2/Normalized_data/Bebop2_16g_1kdps_normalized_0000.csv')
scaler = StandardScaler()
data = scaler.fit_transform(df.values)

data_tensor = torch.tensor(data, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(data_tensor.shape)
