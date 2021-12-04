# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# from utilities import get_test_instances_with_random_samples
# from performance_and_fairness_measures import getHitRatio, getNDCG
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
# import heapq  # for retrieval topK

import torch
import torch.nn as nn
import torch.optim as optim
from torch import LongTensor, FloatTensor

from evaluators import evaluate_model, eval_model
from models import NCF


# %% Hardcode Random State ( 1 )
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


# %% Data Loader Class
class LoadData(Dataset):
    def __getitem__(self, index) -> T_co:
        return self.train_data.uid.iloc[index], self.train_data.pos.iloc[index]

    def __init__(self, data: tuple):
        self.train_data, self.test_data = data
        self.movies = set(self.train_data.pos.unique())
        # self.testing_tensors = self.parse_testing(self.test_data)

    def __len__(self):
        return len(self.train_data)
    @staticmethod
    def parse_testing(df):
        test = df.sort_values(by=['uid', 'rating'], ascending=False)
        users, movies, outputs = [], [], []
        for _, u in test.groupby('uid'):
            users.append(torch.LongTensor(u.uid.to_numpy()))
            # users.append(LongTensor([u.uid.values]))
            movies.append(torch.LongTensor(u.pos.to_numpy()))
            # movies.append(LongTensor([u.mid.values]))
            outputs.append(torch.LongTensor(u.rating.to_numpy()))
            # outputs.append(LongTensor([u.rating.values]))
        return users, movies, outputs

# %% Previous Training function
# def train_epochs(model, df_train, epochs, lr, batch_size, num_negatives, unsqueeze=False):
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
#     model.train()
#     for i in range(epochs):
#         j = 0
#         for batch_i in range(0, np.int64(np.floor(len(df_train) / batch_size)) * batch_size, batch_size):
#             data_batch = (df_train[batch_i:(batch_i + batch_size)]).reset_index(drop=True)
#             # train_user_input, train_item_input, train_ratings = get_instances_with_neg_samples(data_batch, probabilities, num_negatives,device)
#             train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(
#                 data_batch,
#                 num_uniqueLikes,
#                 num_negatives,
#                 device
#             )
#
#             if unsqueeze:
#                 train_ratings = train_ratings.unsqueeze(1)
#             y_hat = model(train_user_input, train_item_input)
#             loss = criterion(y_hat, train_ratings)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print('epoch: ', i, 'batch: ', j, 'out of: ', np.int64(np.floor(len(df_train) / batch_size)),
#                   'average loss: ', loss.item())
#             j = j + 1


# %% Current Trainer
def train_epochs2(model, data, epochs, lr, batch_size, n_negatives, device):
    it_per_epoch = len(data.train_data) / batch_size

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    for i in range(epochs):
        model.train()
        pct, loss = 0, 0
        for batch in DataLoader(data, batch_size, shuffle=False, num_workers=0):

            # batch_index = batch_i * batch_size
            # batch = data.iloc[batch_index: batch_index + batch_size]
            user_input, item_input = batch
            n = len(user_input)  # batch size

            user = torch.zeros((n + n * num_negatives), dtype=torch.long).to(device)
            movie = torch.zeros((n + n * num_negatives), dtype=torch.long).to(device)
            rating = torch.zeros((n + n * num_negatives), dtype=torch.float).to(device)

            for i_u in user_input.unique():  # for each user
                index = 0

                msk = torch.eq(user_input, i_u)
                user_array = user_input[msk]
                item_array = item_input[msk]

                un = len(user_array)  # user array size

                watched_movies = set(item_array.numpy())
                all_movies = ds.movies
                not_seen = all_movies - watched_movies
                # p = torch.ones(len(not_seen)) / len(all_movies)  # equal distribution
                neg_samples = torch.multinomial(FloatTensor(np.array(list(not_seen))), num_samples=un * num_negatives, replacement=True)
                neg_index = 0
                for i_r in range(un):
                    user[index] = i_u
                    movie[index] = item_array[i_r]
                    rating[index] = 1
                    index += 1
                    for s in range(num_negatives):
                        user[index] = i_u
                        movie[index] = neg_samples[neg_index]
                        rating[index] = 0
                        index += 1
                        neg_index += 1

            y_hat = model(LongTensor(user), LongTensor(movie))
            loss = criterion(y_hat, FloatTensor(rating).unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if int(pct / it_per_epoch * 100 % 5) == 0:
                print(f'\rEpoch:{i} -- {int(pct / it_per_epoch * 100)}% -- loss: {loss}', end='')
            pct += 1

        avg_hr_pre_train, avg_ndcg_pre_train = evaluate_model(
            model,
            data.test_data.values,
            top_K,
            random_samples,
            num_unique_likes,
            device
        )
        print(f'\nHr: {avg_hr_pre_train[-1]}\nndcg:{avg_ndcg_pre_train[1]}\n')
        # hr, ndcg = eval_model(model, test_data.values, num_users= 6040, device=device)
        # print(f'\nHr: {hr}\nndcg:{ndcg}\n')

    return evaluate_model(
            model,
            test_data.values,
            top_K,
            random_samples,
            num_unique_likes,
            device
        )


# %% MAIN FUNCTION / HYPER-PARAMETERS
if __name__ == '__main__':
    RANDOM_STATE = 1
    set_random_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # %% set hyperparameters
    emb_size = 128
    hidden_layers = np.array([emb_size, 64, 32, 16])
    output_size = 1
    num_epochs = 25
    learning_rate = 0.001
    batch_size = 2048
    num_negatives = 5
    random_samples = 100
    top_K = 10

    train_data = pd.read_csv('train-test/train_userPages.csv')
    test_data = pd.read_csv('train-test/test_userPages.csv')

    num_uniqueUsers = len(train_data.uid.unique())
    num_unique_likes = len(train_data.pos.unique())
    ds = LoadData((train_data, test_data))

    preTrained_NCF = NCF(num_uniqueUsers, num_unique_likes, emb_size, hidden_layers, output_size).to(
        device)

    hr, ndcg = train_epochs2(preTrained_NCF, ds, num_epochs, learning_rate, batch_size, num_negatives, device)
    torch.save(preTrained_NCF.state_dict(), "trained-models/preTrained_NCF")
    # avg_HR_preTrain, avg_NDCG_preTrain = evaluate_model(preTrained_NCF, test_data.values, top_K, random_samples,
    #                                                     num_uniqueLikes)
    np.savetxt('results/avg_HR_preTrain.txt', hr)
    np.savetxt('results/avg_NDCG_preTrain.txt', ndcg)

    # sys.stdout.close()
