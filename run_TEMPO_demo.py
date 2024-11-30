# Third-party library imports
import numpy as np
import torch
from numpy.random import choice
# Local imports
from tempo.models.TEMPO import TEMPO
from utils.tools import load_data_from_huggingface


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
