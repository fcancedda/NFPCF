import numpy as np
import pandas as pd
import torch

from time import time

from models import NCF
from data import TargetData, AttributeData
from evaluators import eval_model, evaluate_model

d = TargetData()
features = AttributeData()


fd = pd.merge(d.df, features.df, on=['uid', 'uid'], how='right')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------
# %% --- PARSING DATA ---
print("Processing data...")
data = TargetData()
print("Done")

# -----------------------------------------------------------------
# %% --- Hyper parameters ---
num_epochs = 200
batch_size = 128
learning_rate = .05

emb_size = 128  # LATENT DIM

hidden_layers = np.array([emb_size, 64, 32, 16])
output_size = 1

random_samples = 100
num_negatives = 4
top_k = 10

