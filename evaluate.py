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
from torch import LongTensor


class Evaluate:
    def __init__(self, model, data: DataGenerator, k: int = 10, num_thread: int = 8, device: str = 'cuda'):
        # self.data = data
        self.model = model
        self.data = data
        self.test_df = data.add_negatives(data.test, n_samples=100)

        self.k = k
        self.num_thread = num_thread
        self.device = device

    def __call__(self):
        return self.evaluate_model()

    def evaluate_model(self):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        # Single thread
        for idx in range(self.data.num_users):
            (hr, ndcg) = self.eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)
        return np.mean(hits), np.mean(ndcgs)

    def eval_one_rating(self, user):
        items = self.data.testing_tensors[1][user].squeeze()
        user = self.data.testing_tensors[0][user].squeeze()

        true_item = items[0]

        predictions = self.model(user.to(self.device), items.to(self.device))
        map_item_score = dict(zip(items, predictions))
        # for i in range(len(predictions)):
        #     item = items[i]
        #     map_item_score[item] = predictions[i]
        # items.pop()
        hr, ndcg = [], []
        # for i in positive_items:
            # Evaluate top rank list
        ranklist = heapq.nlargest(self.k, map_item_score, key=map_item_score.get)
        hr.append(get_hit_ratio(ranklist, true_item))
        ndcg.append(get_ndcg(ranklist, true_item))
        return np.mean(hr), np.mean(ndcg)

    # def get_test_tensor(self, df_val):
    #     # Prepare the test data points as tensors
    #     test_df = self.data.add_negatives(df_val, n_samples=100)
    #     users, items = LongTensor(test_df.uid).to(self.device), LongTensor(test_df.mid).to(self.device)
    #     # test_negatives = test_neg[100 * (user - 1):100 * user]
    #     return users, items


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
