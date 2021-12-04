import heapq
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import LongTensor, FloatTensor

from fairness_measures import Measures
from models import NCF
from data import LoadData

m = Measures()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% DEBIAS CURRENT USER EMBEDDING
def de_bias_embedding(model, data):
    # LOAD USER EMBEDDING AND WEIGH BY GENDER
    user_embeds = model.user_emb.weight.data.cpu().detach().numpy()
    user_embeds = user_embeds.astype('float')

    ''' COMPUTE GENDER EMBEDDING '''
    gender_embed = np.zeros((2, user_embeds.shape[1]))
    num_users_x_group = np.zeros((2, 1))

    for i in range(data.user_id.shape[0]):
        u = data.user_id.iloc[i]
        if data.gender.iloc[i] == 0:
            gender_embed[0] += user_embeds[u]
            num_users_x_group[0] += 1.0
        else:
            gender_embed[1] += user_embeds[u]
            gender_embed[1] += 1.0
            num_users_x_group[1] += 1.0

    ''' VERTICAL BIAS'''
    gender_embed = gender_embed / num_users_x_group
    # vBias = compute_bias_direction(gender_embed)
    vBias = gender_embed[1].reshape((1, -1)) - gender_embed[0].reshape((1, -1))
    vBias = vBias / np.linalg.norm(vBias, axis=1, keepdims=1)

    ''' LINEAR PROJECTION '''
    debiased_user_embeds = user_embeds
    for i in range(data.user_id.shape[0]):
        u = data.user_id.iloc[i]
        debiased_user_embeds[u] = user_embeds[u] - (np.inner(user_embeds[u].reshape(1, -1), vBias)[0][0]) * vBias

    print(f'males / female {max(num_users_x_group) / min(num_users_x_group)}')
    return torch.from_numpy(debiased_user_embeds.astype(np.float32)).to(device)


# %% FAIRNESS CALCULATOR
def fairness_measures(model, df_val, num_items, protectedAttributes):
    model.eval()
    user_input = torch.LongTensor(df_val['user_id'].values).to(device)
    item_input = torch.LongTensor(df_val['like_id'].values).to(device)
    y_hat = model(user_input, item_input)

    avg_epsilon = m.compute_edf(
        protected_attributes=protectedAttributes,
        predictions=y_hat,
        n_classes=num_items,
        item_input=item_input,
        device=device
    )
    U_abs = m.compute_absolute_unfairness(protectedAttributes, y_hat, num_items, item_input, device=device)

    avg_epsilon = avg_epsilon.cpu().detach().numpy().reshape((-1,)).item()
    print(f"average differential fairness: {avg_epsilon: .3f}")

    U_abs = U_abs.cpu().detach().numpy().reshape((-1,)).item()
    print(f"absolute unfairness: {U_abs: .3f}")


# %% FINE-TUNE FUNCTION
def fine_tune(model, ds, epochs: int = 25, lr: float = .001):
    # train, test = data
    it_per_epoch = len(ds.gender)
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    all_user_input = torch.LongTensor(ds.user_id.values).to(device)
    all_item_input = torch.LongTensor(ds.like_id.values).to(device)
    # ds = LoadData(train, test)
    for i in range(epochs):
        pct = 0
        for batch in DataLoader(ds, shuffle=False, num_workers=0):
            user, item, target = ds.add_negatives(batch, num_negatives, set(ds.like_id.unique()), device)

            y_hat = model(LongTensor(user).to(device), LongTensor(item).to(device))
            loss1 = criterion(y_hat, FloatTensor(target).unsqueeze(1))

            predicted_probs = model(all_user_input, all_item_input)
            avg_epsilon = m.compute_edf(ds.gender, predicted_probs, 17, all_item_input, device)
            loss2 = torch.max(torch.tensor(0.0).to(device), (avg_epsilon - epsilonBase))

            loss = loss1 + fairness_thres * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if int(pct / it_per_epoch * 100 % 5) == 0:
                print(f'\rEpoch:{i} -- {int(pct / it_per_epoch * 100)}% -- loss: {loss}', end='')
            pct += 1
        ds.eval_results(ncf, ds.test, n_items=n_careers, note='Fine-Tuning')


# %% MAIN
if __name__ == "__main__":
    emb_size = 128
    hidden_layers = np.array([emb_size, 64, 32, 16])
    output_size = 1
    num_epochs = 10
    batch_size = 256
    num_negatives = 5
    random_samples = 15
    top_k = 10
    learning_rate = .001
    # LOAD PRE-TRAINED MODEL
    ncf = NCF(6040, 3952, emb_size, hidden_layers, output_size).to(device)
    ncf.load_state_dict(torch.load("models/preTrained_NCF"))
    ds = LoadData()

    # ds.eval_results(ncf, pd.read_csv('train-test/test_userPages.csv'), note='PRE DE-BIAS')

    # FETCH NUMBER OF UNIQUE CAREERS
    n_careers = 17
    '''UPDATE USER EMBEDDINGS'''
    fairness_thres = torch.tensor(0.1).to(device)
    epsilonBase = torch.tensor(0.0).to(device)

    # replace page items with career items
    ncf.like_emb = nn.Embedding(n_careers, emb_size).to(device)
    # freeze user embedding
    ncf.user_emb.weight.requires_grad = False
    # replace user embedding of the model with debiased embeddings

    ncf.user_emb.weight.data = de_bias_embedding(ncf, ds)

    # ds.eval_results(ncf, ds.test, n_items=n_careers, note='POST DE-BIAS')

    fine_tune(ncf, ds, n_careers)
