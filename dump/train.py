import torch
import numpy as np
from time import time
from torch.utils.data import DataLoader

from data import TargetData
from models import NCF
from evaluators import eval_model, evaluate_model

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


# -----------------------------------------------------------------
'''TRAIN MODEL ( MINI BATCH )'''


# %% --- Functions for training ---
def train(model):
    # data.get_train_instances(seed=e)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # privacy_engine.attach(optimizer)

    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0
                            )
    t1 = time()
    # it_per_epoch = len(data) / batch_size
    loss = 0

    for i in range(num_epochs):
        model.train()
        print("Starting epoch ", i + 1)
        # j = 0
        for batch in dataloader:
            u, m, r = batch
            # move tensors to cuda
            u = u.to(device)
            m = m.to(device)
            r = r.to(device)

            y_hat = model(u.squeeze(1), m.squeeze(1))
            loss = torch.nn.BCELoss()  # (weight=w, reduction="mean")

            loss = loss(y_hat, r.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time()
        print("Epoch time:", round(t2 - t1), "seconds")
        print("Loss:", loss.cpu().detach().numpy().round(3))
        model.eval()
        hr, ndcg = eval_model(model, data, data.num_users, device)

        print(f"HR@{top_k}: {round(hr, 2)}  NDCG@{top_k}: {round(ndcg, 2)}\n")
    print("Done")


# %% --- TRAIN MODEL ( MINI BATCH ) ---
ncf = NCF(data.num_users, data.num_movies, emb_size, hidden_layers, output_size).to(device)
# -----------------------------------------------------------------

from opacus import PrivacyEngine

privacy_engine = PrivacyEngine(
    ncf,
    sample_rate=0.01,
    alphas=[10, 100],
    noise_multiplier=1.3,
    max_grad_norm=1.0,
)

# %%
train(ncf)

# %%
# torch.save(ncf.state_dict(), "saved_models/preTrained_NCF")
