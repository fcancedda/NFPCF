import random

import numpy as np
# import utils
import heapq
import math

import pandas as pd
from torch import LongTensor, zeros, long
from pandas import concat, read_csv, DataFrame

'''Eval funcitons'''


def parse_testing(df, n_samples: int = 100):
    unique = df.like_id.unique()
    df['rating'] = np.int8(1)
    combine = df.groupby('user_id')['like_id'].apply(set).reset_index()
    combine['negatives'] = combine['like_id'].apply(lambda x: random.sample(list(set(unique) - x), n_samples))

    s = combine.apply(lambda x: pd.Series(x.negatives, dtype=np.int16), axis=1).stack().reset_index()
    s.rename(columns={'level_0': 'user_id', 0: 'like_id'}, inplace=True)
    s.drop(['level_1'], axis=1, inplace=True)
    s['rating'] = np.int8(0)
    s.user_id = s.uid.astype(np.int16)

    complete = pd.concat([df, s]).sort_values(by=['user_id', 'like_id'])
    complete.reset_index(drop=True, inplace=True)
    test = complete.sort_values(by=['user_id', 'rating'], ascending=False)
    users, movies, outputs = [], [], []
    for _, u in test.groupby('user_id'):
        users.append(LongTensor(u.user_id.to_numpy()))
        movies.append(LongTensor(u.like_id.to_numpy()))
        outputs.append(LongTensor(u.rating.to_numpy()))
    return users, movies, outputs


def eval_results(
        model,
        data: DataFrame = None,
        n_items: int = None,
        n_samples: int = 100,
        note: str = 'Pre-Training {USER, ITEM} -> {RATING}',
        mode: bool = "RATINGS",
        k: int = 10,
        device='cpu'
):
    if mode == "CAREERS":
        data = (
            concat(
                [
                    read_csv("train-test/test_usersID.csv", names=['user_id']),
                    read_csv("train-test/test_concentrationsID.csv", names=['like_id'])
                    # test_protected_attributes['gender']],
                ], axis=1)
        ).reset_index(drop=True)
    elif mode == 'RATINGS':
        data = read_csv('train-test/test_userPages.csv')

    if n_items is None:
        n_items = data.like_id.nunique()

    # results = evaluate_model_old(
    #     model,
    #     data.values,
    #     k,
    #     n_samples,
    #     n_items,
    #     device
    # )
    # print(
    #     f'-- ({note})'
    #     f'\nHr: {results[0][-1]}'
    #     f'\nndcg:{results[1][-1]}\n '
    # )
    results = eval_model(model, parse_testing(data, n_samples=n_samples), num_users=6040, device=device)
    print(
        f'-- ({note})'
        f'\nHr: {results[0]}'
        f'\nndcg:{results[1]}\n '
    )


# %% EVAL FUNC 1
def rank(arr, item):
    # rank of the test item in the list of negative instances
    # returns the number of elements that the test item is bigger than

    index = 0
    for element in arr:
        if element > item:
            index += 1
            return index
        index += 1
    return index


def eval_model(model, testing_tensors, num_users=6040, device='cuda'):
    # Evaluates the model and returns HR@10 and NDCG@10
    hits = 0
    ndcg = 0
    for u in range(num_users):
        user = testing_tensors[0][u].squeeze().to(device)
        item = testing_tensors[1][u].squeeze().to(device)
        y = model(user, item)

        y = y.tolist()
        y = sum(y, [])

        first = y.pop(0)

        y.sort()
        ranking = rank(y, first)
        if ranking > 90:
            hits += 1
            ndcg += np.log(2) / np.log(len(user) - ranking + 1)

    hr = hits / num_users
    ndcg = ndcg / num_users
    return hr, ndcg


# %% EVALUATOR 2 (RETURNS ALL K VALS <= K)
def get_hit_ratio(rank_list, true_item):
    for item in rank_list:
        if item == true_item:
            return 1
    return 0


def get_ndcg(rank_list, true_item):
    for i in range(len(rank_list)):
        item = rank_list[i]
        if item == true_item:
            return math.log(2) / math.log(i + 2)
    return 0


def get_test_instances_with_random_samples(data, random_samples, num_items, device):
    user_input = np.zeros((random_samples + 1))
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
    return LongTensor(user_input).to(device), LongTensor(item_input).to(device)


def evaluate_model_old(model, df_val, top_K, random_samples, num_items, device):
    model.eval()
    avg_HR = np.zeros((len(df_val), top_K))
    avg_NDCG = np.zeros((len(df_val), top_K))

    for i in range(len(df_val)):
        test_user_input, test_item_input = get_test_instances_with_random_samples(
            df_val[i],
            random_samples,
            num_items,
            device
        )
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
            avg_HR[i, k] = get_hit_ratio(ranklist, gtItem)
            avg_NDCG[i, k] = get_ndcg(ranklist, gtItem)
    avg_HR = np.mean(avg_HR, axis=0)
    avg_NDCG = np.mean(avg_NDCG, axis=0)
    return avg_HR, avg_NDCG


# %% NIGHT BUILD ;)
def evaluate_model(model, df_val, top_k, random_samples, num_items, device):
    model.eval()
    avg_hr = np.zeros((len(df_val), top_k))
    avg_ndcg = np.zeros((len(df_val), top_k))

    for i in range(len(df_val)):
        user, target = df_val[i, 0], df_val[i, 1]
        user_input = zeros((random_samples + 1), dtype=long).to(device)
        item_input = zeros((random_samples + 1), dtype=long).to(device)

        index = 0
        # negative instances
        for t in range(random_samples):
            j = np.random.randint(num_items)
            while j == target:
                j = np.random.randint(num_items)
            user_input[index] = user
            item_input[index] = j
            index += 1
        # positive instance
        user_input[index] = user
        item_input[index] = target

        y_hat = model(LongTensor(user_input).to(device), LongTensor(item_input).to(device))
        y_hat = y_hat.cpu().detach().numpy().reshape((-1,))
        test_item_input = item_input.cpu().detach().numpy().reshape((-1,))
        map_item_score = {}
        for j in range(1, len(y_hat)):
            map_item_score[test_item_input[j]] = y_hat[j]

        # ADD original at end
        map_item_score[test_item_input[0]] = y_hat[0]

        for k in range(top_k):
            # Evaluate top rank list
            ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
            avg_hr[i, k] = get_hit_ratio(ranklist, target)
            avg_ndcg[i, k] = get_ndcg(ranklist, target)

    avg_hr = np.mean(avg_hr, axis=0)
    avg_ndcg = np.mean(avg_ndcg, axis=0)
    return avg_hr, avg_ndcg
