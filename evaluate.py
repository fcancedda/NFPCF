'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from data import DataGenerator
from pandas import DataFrame
from torch import LongTensor


class Evaluate:
    def __init__(self, data: DataGenerator, testing_data: DataFrame = None, k: int = 10, num_thread: int = 8, device: str = 'cuda'):
        self.data = data
        if testing_data is None:
            testing_data = data.test
        self.test_df = data.add_negatives(testing_data, n_samples=100)

        self.k = k
        self.num_thread = num_thread
        self.device = device

    def __call__(self, model):
        return self.evaluate_model(model)

    def evaluate_model(self, model):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        # Single thread
        for idx in range(self.data.num_users):
            (hr, ndcg) = self.eval_one_rating(model, idx)
            hits.append(hr)
            ndcgs.append(ndcg)
        return np.mean(hits), np.mean(ndcgs)

    def eval_one_rating(self, model, user):
        items = self.data.testing_tensors[1][user].squeeze()
        user = self.data.testing_tensors[0][user].squeeze()

        true_item = items[0]

        predictions = model(user.to(self.device), items.to(self.device))
        map_item_score = dict(zip(items, predictions))
        hr, ndcg = [], []
        ranklist = heapq.nlargest(self.k, map_item_score, key=map_item_score.get)
        hr.append(get_hit_ratio(ranklist, true_item))
        ndcg.append(get_ndcg(ranklist, true_item))
        return np.mean(hr), np.mean(ndcg)


def get_hit_ratio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def get_ndcg(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0
