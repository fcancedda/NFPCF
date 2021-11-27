import torch
import numpy as np
from time import time
from torch.utils.data import DataLoader

from data import TargetData, MovieTrainData
from models import NCF3
from evaluators import eval_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------
# %% --- Hyper parameters ---
num_epochs = 25
batch_size = 128
learning_rate = .01

emb_size = 128  # LATENT DIM
hidden_layers = np.array([emb_size, 64, 32, 16])
output_size = 1

random_samples = 100
num_negatives = 4
top_k = 10
# loss_function = torch.nn.BCEWithLogitsLoss()
loss_function = torch.nn.BCELoss()  # (weight=w, reduction="mean")


# -----------------------------------------------------------------
# %% --- Functions for training ---
def train(model, movie_data):
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    dataloader = DataLoader(MovieTrainData(movie_data.training_data), batch_size=128, shuffle=True, num_workers=1)
    t1 = time()
    loss = 0
    it_per_epoch = len(movie_data) / batch_size
    for i in range(num_epochs):
        model.train()
        j = 0
        for batch in dataloader:
            u, m, r = batch
            # move tensors to cuda
            u = u.to(device)
            m = m.to(device)
            r = r.to(device)

            y_hat = model(u.squeeze(1), m.squeeze(1))
            loss = loss_function(y_hat, r.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % int(1 + it_per_epoch / 10) == 0:
                print(f"\r Epoch {i + 1}, Progress: {round(100 * j / it_per_epoch)}%", end='', flush=True)
            j += 1

        t2 = time()
        print("\nEpoch time:", round(t2 - t1), "seconds")
        print("Loss:", loss.cpu().detach().numpy())
        model.eval()
        hr, ndcg = eval_model(model, data, num_users=data.num_users, device=device)
        print(f"HR@{top_k}: {hr}  NDCG@{top_k}: {round(ndcg, 2)}\n")
    print("Done")


# -----------------------------------------------------------------
# %% --- TRAIN MODEL ( MINI BATCH ) ---
if __name__ == '__main__':
    # %% --- PARSING DATA ---
    print("Processing data...")
    data = TargetData()
    print("Done")
    # %% --- LOAD MODEL AND TRAIN ---
    ncf = NCF3(
        data.num_users,
        data.num_movies,
        emb_size,
        num_layers=4,
        dropout=1
    ).to(device)

    train(ncf, data)

    # %% --- SAVE MODEL ---
    torch.save(ncf.state_dict(), "saved_models/NCF")
    print('Model Saved')
