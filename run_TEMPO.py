# Standard library imports
import argparse
import os
import random
import sys
import time
import warnings

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.random import choice
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import Subset
from tqdm import tqdm

# Local imports
from data_provider.data_factory import data_provider
from models.TEMPO import TEMPO
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test, load_data_from_huggingface

def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config

# import subprocess

# # Define the command as a list of arguments
# command = [
#     "huggingface-cli", "download",
#     "Melady/TEMPO", "all_six_datasets/pems-bay.csv",
#     "--local-dir", "./datasets/"
# ]

# # Execute the command
# result = subprocess.run(command, capture_output=True, text=True)

# # Check if the command was successful
# if result.returncode == 0:
#     print("Download successful")
# else:
#     print("Download failed")
#     print("Error message:", result.stderr)

# pems_bay = pd.read_csv('./datasets/all_six_datasets/pems-bay.csv')

# Usage
repo_id = "Melady/TEMPO"
filename = "all_six_datasets/pems-bay.csv"

pems_bay = load_data_from_huggingface(repo_id, filename)

if pems_bay is not None:
    # You can now use pems_bay DataFrame directly
    print(pems_bay.head())
    print(f"Shape of the dataset: {pems_bay.shape}")

config = {
    "description": "TEMPO",
    "model_id": "TEMPO_checkpoints/etth2_336_96",
    "checkpoints": "./checkpoints/",
    "task_name": "long_term_forecast",
    "prompt": 1,
    "num_nodes": 1,
    "seq_len": 336,
    "pred_len": 96,
    "label_len": 96,
    "decay_fac": 0.5,
    "learning_rate": 0.001,
    "batch_size": 256,
    "num_workers": 0,
    "train_epochs": 10,
    "lradj": "type3",
    "patience": 5,
    "gpt_layers": 6,
    "is_gpt": 1,
    "e_layers": 3,
    "d_model": 768,
    "n_heads": 4,
    "d_ff": 768,
    "dropout": 0.3,
    "enc_in": 7,
    "c_out": 1,
    "patch_size": 16,
    "kernel_size": 25,
    "loss_func": "mse",
    "pretrain": 1,
    "freeze": 1,
    "model": "TEMPO",
    "stride": 8,
    "max_len": -1,
    "hid_dim": 16,
    "tmax": 20,
    "itr": 3,
    "cos": 1,
    "equal": 1,
    "pool": False,
    "no_stl_loss": False,
    "stl_weight": 0.001,
    "config_path": "./configs/custom_datasets.yml",
    "datasets": "ETTm1,ETTh1,ETTm2,electricity,traffic,weather",
    "target_data": "Custom",
    "use_token": 0,
    "electri_multiplier": 1,
    "traffic_multiplier": 1,
    "embed": "timeF",
    "percent": 100,
}

if not os.path.exists("./configs"):
    os.makedirs("./configs")
#./configs/run_TEMPO.yml
if not os.path.exists("./configs/run_TEMPO.yml"):
    with open("./configs/run_TEMPO.yml", "w") as f:
        print("Configuration written to config.yml")
        cfg = OmegaConf.create(config)
        OmegaConf.save(config, f)
        
else:
    print("Configuration file already exists")
    cfg = OmegaConf.load("./configs/run_TEMPO.yml")


# if CPU is used, the device is set to 'cpu', otherwise, the device is set to 'cuda:0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = TEMPO.load_pretrained_model(cfg, device)

print("Successfully loaded the model")


def prepare_data_and_predict(model, df, seq_length=336, pred_length=96, total_length=432):
    # Randomly select a column (excluding the first column)
    selected_column = random.choice(df.columns[1:])
    
    # Get the total length of the selected column
    column_length = len(df[selected_column])
    
    # Randomly select a starting point, ensuring we have enough data
    max_start = column_length - total_length
    start_idx = random.randint(0, max_start)
    
    # Extract 432 values from the selected column, starting from the random start point
    data = df[selected_column].values[start_idx:start_idx+total_length]
    
    # Prepare input for the model (first 336 values)
    input_data = data[:seq_length]
    
    # Get prediction
    with torch.no_grad():
        prediction = model.predict(input_data)
    
    # Extract the predicted values (last 96 values)
    predicted_values = prediction.squeeze().numpy()[-pred_length:]
    
    return input_data, predicted_values, data[seq_length:], selected_column, start_idx

def plot_results(input_data, predicted_values, actual_values, column_name, start_idx):
    plt.figure(figsize=(15, 6))
    
    # Plot input data
    plt.plot(range(start_idx, start_idx + len(input_data)), input_data, label='Input Data', color='blue')
    
    # Plot predicted values
    plt.plot(range(start_idx + len(input_data), start_idx + len(input_data) + len(predicted_values)), 
             predicted_values, label='Predicted', color='red')
    
    # Plot actual values
    plt.plot(range(start_idx + len(input_data), start_idx + len(input_data) + len(actual_values)), 
             actual_values, label='Actual', color='green')
    
    plt.title(f'TEMPO Prediction for {column_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results_results/{column_name}_{start_idx}_prediction.png')

    # plt.show()

    # Prepare data and get prediction
for i in range(10):
    input_data, predicted_values, actual_values, selected_column, start_idx = prepare_data_and_predict(model, pems_bay)

    # Plot the results
    plot_results(input_data, predicted_values, actual_values, selected_column, start_idx)

    # Print some statistics
    print(f"Selected Column: {selected_column}")
    print(f"Start Index: {start_idx}")
    print(f"Mean Absolute Error: {np.mean(np.abs(predicted_values - actual_values))}")
    print(f"Root Mean Squared Error: {np.sqrt(np.mean((predicted_values - actual_values)**2))}")