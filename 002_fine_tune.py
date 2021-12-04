import heapq
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import LongTensor, FloatTensor

from fairness_measures import Measures
from models import NCF
import utils
import data

m = Measures()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% DEBIAS CURRENT USER EMBEDDING
def de_bias_embedding(model, data):
    uid, gender, job = data
    # LOAD USER EMBEDDING AND WEIGH BY GENDER
    user_embeds = model.user_emb.weight.data.cpu().detach().numpy()
    user_embeds = user_embeds.astype('float')

    ''' COMPUTE GENDER EMBEDDING '''
    gender_embed = np.zeros((2, user_embeds.shape[1]))
    num_users_x_group = np.zeros((2, 1))

    for i in range(uid.shape[0]):
        u = uid.iloc[i]
        if gender.iloc[i] == 0:
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
    for i in range(uid):
        u = uid.iloc[i]
        debiased_user_embeds[u] = user_embeds[u] - (np.inner(user_embeds[u].reshape(1, -1), vBias)[0][0]) * vBias

    print(num_users_x_group)
    print(vBias)
    print(gender_embed)
    return torch.from_numpy(debiased_user_embeds.astype(np.float32)).to(device)


class LoadData(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.user_id = df.user_id
        self.like_id = df.like_id
        self.gender = df.gender

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.user_id.iloc[index],
            self.like_id.iloc[index],
            self.gender.iloc[index]
        )


def add_negatives(batch: (), num_negatives, movies: {}):
    # data_batch = (df_train[batch_i:(batch_i + batch_size)]).reset_index(drop=True)
    user_input, item_input = batch
    n = len(user_input)  # ~=batch size
    n += n * num_negatives

    user = torch.zeros(n, dtype=torch.long).to(device)
    movie = torch.zeros(n, dtype=torch.long).to(device)

    rating = torch.zeros(n, dtype=torch.float).to(device)

    for i_u in user_input.unique():  # for each user
        index = 0

        msk = torch.eq(user_input, i_u)
        user_array = user_input[msk]
        item_array = item_input[msk]

        un = len(user_array)  # user array size

        watched_movies = set(item_array.numpy())
        not_seen = movies - watched_movies
        # p = torch.ones(len(not_seen)) / len(all_movies)  # equal distribution
        neg_samples = torch.multinomial(FloatTensor(np.array(list(not_seen))), num_samples=un * num_negatives,
                                        replacement=True)
        neg_index = 0
        for i_r in range(un):
            user[index] = i_u
            movie[index] = item_array[i_r]
            rating[index] = 1
            index += 1
            for s in range(num_negatives):
                user[index] = i_u
                movie[index] = neg_samples[neg_index]
                # rating[index] = 0
                index += 1
                neg_index += 1
    return user, movie, rating


# %% EVALUATOR FUNCTION
def evaluate_fine_tune(model, df_val, top_K, random_samples, num_items):
    model.eval()
    avg_HR = np.zeros((len(df_val), top_K))
    avg_NDCG = np.zeros((len(df_val), top_K))

    for i in range(len(df_val)):
        test_user_input, test_item_input = user_input = np.zeros((random_samples + 1))
        item_input = np.zeros((random_samples + 1))

        # positive instance
        user_input[0] = data[0]
        item_input[0] = data[1]
        i = 1
        # negative instances
        checkList = data[1]
        for t in range(random_samples):
            j = np.random.randint(num_items)
            while j == checkList:
                j = np.random.randint(num_items)
            user_input[i] = data[0]
            item_input[i] = j
            i += 1

        y_hat = model(test_user_input, test_item_input)
        y_hat = y_hat.cpu().detach().numpy().reshape((-1,))
        test_item_input = test_item_input.cpu().detach().numpy().reshape((-1,))
        map_item_score = {}
        for j in range(len(y_hat)):
            map_item_score[test_item_input[j]] = y_hat[j]
        for k in range(top_K):
            # Evaluate top rank list
            ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
            gtItem = test_item_input[0]
            avg_HR[i, k] = utils.get_hit_ratio(ranklist, gtItem)
            avg_NDCG[i, k] = utils.get_ndcg(ranklist, gtItem)
    avg_HR = np.mean(avg_HR, axis=0)
    avg_NDCG = np.mean(avg_NDCG, axis=0)
    return avg_HR, avg_NDCG


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


def fine_tune(model, train, epochs: int = 25, lr: float = .001):
    it_per_epoch = len(train.gender) / batch_size
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    all_user_input = torch.LongTensor(train.user_id.values).to(device)
    all_item_input = torch.LongTensor(train.like_id.values).to(device)
    ds = LoadData(train)
    for i in range(epochs):
        pct = 0
        for batch in DataLoader(ds, batch_size, shuffle=False, num_workers=0):
            user, item, target = add_negatives(batch, num_negatives, set(all_item_input.unique().numpy()))
            print(item)
            y_hat = model(LongTensor(user).to(device), LongTensor(item).to(device))
            loss1 = criterion(y_hat, FloatTensor(target).unsqueeze(1))

            predicted_probs = model(all_user_input, all_item_input)
            avg_epsilon = m.compute_edf(train.gender, predicted_probs, train.like_id.nunique(), all_item_input, device)
            loss2 = torch.max(torch.tensor(0.0).to(device), (avg_epsilon - epsilonBase))

            loss = loss1 + fairness_thres * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if int(pct / it_per_epoch * 100 % 5) == 0:
                print(f'\rEpoch:{i} -- {int(pct / it_per_epoch * 100)}% -- loss: {loss}', end='')
            pct += 1


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
    # %% load data
    train_users = pd.read_csv("train-test/train_usersID.csv", names=['user_id'])
    test_users = pd.read_csv("train-test/test_usersID.csv", names=['user_id'])

    train_careers = pd.read_csv("train-test/train_concentrationsID.csv", names=['like_id'])
    test_careers = pd.read_csv("train-test/test_concentrationsID.csv", names=['like_id'])

    train_protected_attributes = pd.read_csv("train-test/train_protectedAttributes.csv")
    test_protected_attributes = pd.read_csv("train-test/test_protectedAttributes.csv")

    train_data = (pd.concat(
        [
            train_users['user_id'],
            train_careers['like_id'],
            train_protected_attributes['gender']],
        axis=1
    )).reset_index(drop=True)
    test_data = (pd.concat(
        [
            test_users['user_id'],
            test_careers['like_id'],
            test_protected_attributes['gender']],
        axis=1
    )).reset_index(drop=True)

    # FETCH NUMBER OF UNIQUE CAREERS
    n_careers = 16
    '''UPDATE USER EMBEDDINGS'''
    fairness_thres = torch.tensor(0.1).to(device)
    epsilonBase = torch.tensor(0.0).to(device)

    # replace page items with career items
    ncf.like_emb = nn.Embedding(n_careers, emb_size).to(device)
    # freeze user embedding
    ncf.user_emb.weight.requires_grad = False
    # replace user embedding of the model with debiased embeddings

    ncf.user_emb.weight.data = de_bias_embedding(ncf, train_data)

    fine_tune(ncf, train_data, n_careers)
