import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import glob


file_pattern = '/workspace/UAV_measurement_data/FFT/Bebop2_16g_1kdps_normalized_0022.csv'

training_data=pd.read_csv(file_pattern)


hist1, bins1= np.histogram(training_data['C_aX'], bins=100)

print('C')
print(bins1.mean())

hist2, bins2= np.histogram(training_data['D_aX'], bins=100)

print('D')
print(bins2.mean())


# Plot histogram
plt.hist(training_data['A_aX'], label='A=1', alpha=0.25, bins=100)
plt.hist(training_data['C_aX'], label='B=0', alpha=0.25, bins=100)
plt.hist(training_data['D_aX'], label='D=2', alpha=0.25, bins=100)

plt.legend()
plt.title('Histogram w zaleności od typu uszkodzenia')
plt.xlabel('Value')
plt.ylabel('Data')

# Save the plot
output_path = './temp.png'
plt.savefig(output_path)
plt.close()