from numpy.random import choice, seed
from pandas import read_csv
from torch.utils.data import TensorDataset
from torch import LongTensor
import torch


# %% Hardcode Random State ( 1 )
def set_seed(state: int = 1):
    gens = (seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


def load_data(pre_train: bool = True, device: torch.device = None):
    if pre_train:
        train_data = read_csv('train-test/train_userPages.csv')
        uid = LongTensor(train_data['user_id'].values).to(device)
        mid = LongTensor(train_data['like_id'].values).to(device)
        inps = torch.stack([uid, mid], 1)
    else:
        train_users = read_csv("train-test/train_usersID.csv", names=['user_id'])
        uid = LongTensor(train_users['user_id'].values).to(device)

        train_careers = read_csv("train-test/train_concentrationsID.csv", names=['like_id'])
        mid = LongTensor(train_careers['like_id'].values).to(device)

        train_protected_attributes = read_csv("train-test/train_protectedAttributes.csv")
        gender = LongTensor(train_protected_attributes['gender'].values).to(device)
        inps = torch.stack([uid, mid, gender], 1)
    tgts = torch.ones(len(uid), dtype=torch.float).to(device)
    return TensorDataset(inps, tgts)


# %% ADD NEGATIVE OCCURRENCES FUNCTION (IMPLICIT FEEDBACK)
def add_false(batch: list, n_false: int = 15, n_items: int = 3952, device: torch.device = 'cpu'):
    inputs, targets = batch
    users, items = inputs[:, 0], inputs[:, 1]

    n = set(range(n_items))

    not_seen = {}
    negatives = []

    for user in users.cpu().numpy():
        if user not in not_seen:
            # not_seen[user] = list(n - set(items[users == user].cpu().numpy()))
            not_seen[user] = list(n - set(torch.masked_select(items, torch.eq(users, user)).cpu().numpy()))
        negatives.append(
            (
                torch.full((n_false,), user).to(device),
                torch.tensor(choice(not_seen[user], size=n_false)).to(device),
                torch.zeros(n_false, ).to(device)
            )
        )
    for negative in negatives:
        uu, mm, rr = negative
        users = torch.cat([users, uu])
        items = torch.cat([items, mm])
        targets = torch.cat([targets, rr])
    return users, items, targets
