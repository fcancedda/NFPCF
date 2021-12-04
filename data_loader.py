from torch import LongTensor
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from random import sample
from time import time


# class LoadData(Dataset):
#     def __init__(self):
#         t1 = time()
#         self.ratings = self._load_ratings()
#         t2 = time()
#         self.user_features = self._load_features()
#         t3 = time()
#         self.fd = pd.merge(self.ratings, self.user_features, on=['uid', 'uid'], how='right')
#         t4 = time()
#         self.fd = self._reset_index(self.fd, ['uid', 'mid', 'job'])
#         print(self.fd.head())
#         self.train, self.test = self._train_test_split()
#         movies = set(self.fd.mid.unique())
#         self.tr = self.add_negatives(self.train, items = movies, n_samples=5)
#         self.te = self.add_negatives(self.test, items= movies, tr=self.train, n_samples=100)
#         self.testing_tensors = self.parse_testing(self.te)
#
#         t6 = time()
#         print('ratings', t2 - t1)
#         print('features', t3 - t2)
#         print('merge', t4 - t3)
#         print('negatives', t6 - t4)
#         print('total',t6-t1)
#
#     def __len__(self):
#         return self.tr.shape[0]
#
#     def __getitem__(self, item):
#         u, m, r = self.tr[['uid', 'mid', 'rating']].iloc[item]
#         return LongTensor([u]), LongTensor([m]), LongTensor([r])
#
#     @staticmethod
#     def add_negatives(df: pd.DataFrame, item: str = 'mid', items=None, tr = None, n_samples: int = 4):
#         if items is None:
#             items = set(df[item].unique())
#
#         def user_movies(uid, mid):
#             train_movies = set()
#             if tr is not None:
#                 train_movies = set(tr[tr.uid==uid].pos.unique())
#                 if not train_movies:
#                     print(' no movies in train found', uid)
#             return sample(list(items - train_movies - mid), n_samples)
#
#         df['rating'] = np.int8(1)
#         combine = df.groupby('uid')[item].apply(set).reset_index()
#         combine['negatives'] = combine.apply(lambda x: user_movies(x.uid, x.pos), axis=1)
#
#         s = combine.apply(lambda x: pd.Series(x.negatives, dtype=np.int16), axis=1).stack().reset_index()
#         s.rename(columns={'level_0': 'uid', 0: item}, inplace=True)
#         s.drop(['level_1'], axis=1, inplace=True)
#         s['rating'] = np.int8(0)
#         s.uid = s.uid.astype(np.int16)
#
#         # complete = pd.merge([df, s], how='left', on='uid').sort_values(by=['uid', item])
#         complete = pd.concat([df, s]).sort_values(by=['uid', item])
#         return complete.reset_index(drop=True)
#
#     def _train_test_split(self):
#         self.fd['latest'] = self.fd.groupby(['uid'])['date'].rank(method='first', ascending=False)
#         test_bool = self.fd.latest <= 1
#         test = self.fd[test_bool]
#         train = self.fd[~test_bool]
#         return (
#             train[['uid', 'mid']], test[['uid', 'mid']]
#         )
#
#     @staticmethod
#     def _reset_index(df, cols):
#         for col in cols:
#             old_col = col + '_old'
#             df.rename(columns={col: old_col}, inplace=True)
#             user_id = df[[old_col]].drop_duplicates().reindex()
#             user_id[col] = np.arange(len(user_id))
#             df = pd.merge(df, user_id, on=[old_col], how='left')
#         return df
#
#
#     @staticmethod
#     def _load_ratings(min_ratings=5):
#         df = pd.read_csv('MovieLens/ratings.dat',
#                          sep='::',
#                          header=None,
#                          names=['uid', 'mid', 'rating', 'date'],
#                          parse_dates=['date'],
#                          date_parser=lambda x: pd.to_datetime(x, unit='s', origin='unix'),
#                          engine='python')
#
#         # DROP MOVIES WITH LESS THAN 5 RATINGS
#         s = df.groupby(['mid']).size()
#         low_n_ratings = s[s < min_ratings].reset_index().pos.tolist()
#         return df[~df.pos.isin(low_n_ratings)]
#
#
#     @staticmethod
#     def _load_features():
#         df = pd.read_csv('MovieLens/users.dat',
#                          sep='::',
#                          header=None,
#                          names=['uid', 'gender', 'age', 'job', 'zip'],
#                          engine='python')
#         df.gender = pd.get_dummies(df.gender, drop_first=True)  # 0:F, 1:M
#         drop = [0, 10, 13, 19]
#
#         clean = df[~df['job'].isin(drop)]
#         return clean.drop(columns=['zip'])
#
#     @staticmethod
#     def parse_testing(df):
#         test = df.sort_values(by=['uid', 'rating'], ascending=False)
#         users, movies, outputs = [], [], []
#         for _, u in test.groupby('uid'):
#             users.append(LongTensor(u.uid.to_numpy()))
#             # users.append(LongTensor([u.uid.values]))
#             movies.append(LongTensor(u.pos.to_numpy()))
#             # movies.append(LongTensor([u.mid.values]))
#             outputs.append(LongTensor(u.rating.to_numpy()))
#             # outputs.append(LongTensor([u.rating.values]))
#         return users, movies, outputs

