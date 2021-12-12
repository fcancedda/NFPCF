# -*- coding: utf-8 -*-
from numpy import array

import torch
from torch.utils.data import DataLoader

from data import load_data, add_false, set_seed
from evaluators import eval_results
from models import NCF

from time import time

# SET DEVICE STATE (NVIDIA CARD IS OPTIMAL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% TRAINING FUNCTION
def train_model(model):
    loss_ = torch.nn.BCELoss()  # Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    path = "model.pt"
    # LOAD CHECKPOINT
    # checkpoint = torch.load('model.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss_ = checkpoint['loss']
    for i in range(epoch):
        t1 = time()
        model.train()
        pct = 0
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            # pin_memory=True,
            # num_workers=2
        )
        n = len(dataloader)
        for batch in dataloader:
            usr, itm, rtn = add_false(batch, n_false=amount, n_items=unique_items, device=device)
            y_hat = model(usr.to(device), itm.to(device))
            loss = loss_(y_hat, rtn.unsqueeze(1).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if int(pct / n) % 5 == 0:
                print(f'\rEpoch:{i} -- '
                      f'{int(pct / n)}% -- '
                      f'LOSS: {torch.round(loss * 1000) / 1000} -- '
                      f'({int(time() - t1)}s)', end='')
                if int(pct / n) % 50 == 48:  # SAVE CHECKPOINT
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, path)
            pct += 100
        eval_results(model=model, n_samples=random_samples, n_items=unique_items, note='Pre-Training Model on Ratings',
                     device=device)


# %% MAIN FUNCTION / HYPER-PARAMETERS
if __name__ == '__main__':
    RANDOM_STATE = 1
    set_seed()  # FIX RANDOM STATE FOR REPRODUCIBILITY

    emb_size = 128  # MODEL EMBEDDING SIZE
    hidden_layers = array([emb_size, 64, 32, 16])  # NUMBER OF LAYERS
    output_size = 1  # PROBABILITY VALUE BETWEEN 0 AND 1 USE WILL LIKE MOVIE

    epoch = 25  # NUMBER OF ITERATIONS
    learning_rate = 0.001  # LEARNING RATE FOR ADAM OPTIMIZER
    batch_size = 2048  # BATCH SIZE FOR TRAINING
    amount = 5  # AMOUNT OF NEGATIVES PER POSITIVE (TRAIN)
    random_samples = 100  # COMPARING TARGET WITH # SAMPLES (EVAL)
    top_K = 10  # TARGET SCORE MUST BE BETWEEN TOP K ITEMS

    #  LOAD RATINGS DATASET
    dataset = load_data(pre_train=True, device=device)
    unique_users = len(dataset.tensors[0][:, 0].unique())
    unique_items = len(dataset.tensors[0][:, 1].unique())

    # COMPILE MODEL
    # ncf = NCF3(unique_users, unique_items, emb_size, 4, output_size).to(device)
    ncf = NCF(unique_users, unique_items, emb_size, hidden_layers, output_size).to(device)

    train_model(ncf)  # TRAIN MODEL

    torch.save(ncf.state_dict(), "output/preTrained_NCF")  # SAVE MODEL
