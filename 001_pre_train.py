# -*- coding: utf-8 -*-
from numpy import array
from numpy.random import seed

import torch
from torch.utils.data import DataLoader

from data import load_data, add_false
from evaluators import eval_results
from models import NCF3


# %% Hardcode Random State ( 1 )
def set_random_seed(state: int = 1):
    gens = (seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


# %% TRAINING FUNCTION
def train_model(model):
    path = "model.pt"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2
    )

    loss_ = torch.nn.BCELoss()  # Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # LOAD PREVIOUS MODEL
    # checkpoint = torch.load('model.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    n = len(dataloader)
    for i in range(epoch):
        model.train()
        pct = 0
        for batch in dataloader:
            usr, itm, rtn = add_false(batch, amount, n, device)
            y_hat = model(usr, itm)

            loss = loss_(y_hat, rtn.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if int(pct / n) % 5 == 0:
                print(f'\rEpoch:{i} -- {int(pct / n)}% -- loss: {loss}', end='')
                if int(pct / n) % 48 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, path)
            pct += 100
        eval_results(model=model, n_items=unique_items, note='Pre-Training Model on Ratings')


# %% MAIN FUNCTION / HYPER-PARAMETERS
if __name__ == '__main__':
    RANDOM_STATE = 1
    set_random_seed()
    # SET DEVICE STATE (NVIDIA CARD IS OPTIMAL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_size = 128  # MODEL EMBEDDING SIZE
    hidden_layers = array([emb_size, 64, 32, 16])  # NUMBER OF LAYERS
    output_size = 1  # PROBABILITY VALUE BETWEEN 0 AND 1 USE WILL LIKE MOVIE

    epoch = 25  # NUMBER OF ITERATIONS
    learning_rate = 0.001  # LEARNING RATE FOR ADAM OPTIM
    batch_size = 2048  # BATCH SIZE FOR TRAINING
    amount = 5  # AMOUNT OF NEGATIVES PER POSITIVE (TRAIN)
    random_samples = 100  # COMPARING TARGET WITH # SAMPLES (EVAL)
    top_K = 10  # TARGET SCORE MUST BE BETWEEN TOP K ITEMS

    #  LOAD RATINGS DATASET
    dataset = load_data(pre_train=True, device=device)
    unique_users = len(dataset.tensors[0][:, 0].unique())
    unique_items = len(dataset.tensors[0][:, 1].unique())

    # COMPILE MODEL
    ncf = NCF3(unique_users, unique_items, emb_size, 4, output_size).to(device)
    # ncf = NCF(unique_users, unique_items, emb_size, hidden_layers, output_size).to(device)

    # TRAIN MODEL
    train_model(ncf)

    # SAVE MODEL
    # torch.save(preTrained_NCF.state_dict(), "output/preTrained_NCF")
