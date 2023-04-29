import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.getcwd()
path = os.path.join(current_dir, 'Data')
fake_path = os.path.join(path, 'Fake.csv')
true_path = os.path.join(path, 'True.csv')
data_path = os.path.join(path, 'Full_data.csv')

# Read the input CSV files into pandas dataframes
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

# Create a target variable
fake_df['target'] = 0
true_df['target'] = 1

# Combine the dataframes into a single dataframe
combined_df = pd.concat([fake_df, true_df])

# Display the new Dataset
print(combined_df.head())
print('Dataframe columns: ', combined_df.columns)

# Write the combined dataframe to a new CSV file
combined_df.to_csv(data_path, index=False)

print("Combined CSV file saved successfully at", data_path)