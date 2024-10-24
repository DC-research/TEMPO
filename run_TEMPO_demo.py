# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import choice
from omegaconf import OmegaConf
# Local imports
from models.TEMPO import TEMPO
from utils.tools import load_data_from_huggingface

def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config

# Usage
repo_id = "Melady/TEMPO"
filename = "all_six_datasets/pems-bay.csv"

pems_bay = load_data_from_huggingface(repo_id, filename)

if pems_bay is not None:
    # You can now use pems_bay DataFrame directly
    print(pems_bay.head())
    print(f"Shape of the dataset: {pems_bay.shape}")


# # Load the configuration file
# cfg = OmegaConf.load("./configs/run_TEMPO.yml")


model = TEMPO.load_pretrained_model(
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        repo_id = "Melady/TEMPO",
        filename = "TEMPO-80M_v1.pth",
        cache_dir = "./checkpoints/TEMPO_checkpoints"  
)
     
print("Successfully loaded the model")

input_data = np.random.rand(336)    # Random input data
with torch.no_grad():
        predicted_values = model.predict(input_data, pred_length=96)
print("Predicted values:")
print(predicted_values)

############################################################################################################
