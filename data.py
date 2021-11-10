import pandas as pd
import numpy as np
from random import sample
# import tensorflow as tf

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from torch import LongTensor


class DataGenerator(Dataset):
    def __init__(self, num_negatives=4):
        self.num_jobs = 0
        self.df = self.load_ratings()
        self.num_users = self.df.uid.nunique()
        self.num_movies = self.df.mid.nunique()

        self.users = set(self.df.uid.unique())
        self.movies = set(self.df.mid.unique())

        self.train, self.test = self._train_test_split()

        self.training_data = self.add_negatives(self.train, n_samples=num_negatives)

    def __len__(self):
        return self.train.shape[0]  # Length of the data to train on

    def __getitem__(self, index) -> T_co:
        user = LongTensor([self.training_data.uid.iloc[index]])  # -1 so that indexing starts from 0
        movie = LongTensor([self.training_data.mid.iloc[index]])
        output = LongTensor([self.training_data.rating.iloc[index]])
        return user, movie, output

    def _train_test_split(self):
        self.df.rating = np.int8(1)
        self.df['latest'] = self.df.groupby(['uid'])['date'].rank(method='first', ascending=False)
        test_bool = self.df.latest <= 2
        test = self.df[test_bool]
        train = self.df[~test_bool]
        return (train[['uid', 'mid', 'rating']],
                test[['uid', 'mid', 'rating']]
                )

    def add_negatives(self, df: pd.DataFrame, item: str = 'mid', items=None, n_samples: int = 4):
        if items is None:
            items = set(self.df[item].unique())

        combine = df.groupby('uid')[item].apply(set).reset_index()
        combine['negatives'] = combine[item].apply(lambda x: sample(list(items - x), n_samples))

        s = combine.apply(lambda x: pd.Series(x.negatives, dtype=np.int16), axis=1).stack().reset_index()
        s.rename(columns={'level_0': 'uid', 0: item}, inplace=True)
        s.drop(['level_1'], axis=1, inplace=True)
        s['rating'] = np.int8(0)
        s.uid = s.uid.astype(np.int16)

        complete = pd.concat([df, s]).sort_values(by=['uid', item])
        return complete.reset_index(drop=True)

    @staticmethod
    def load_ratings(min_ratings=5):
        df = pd.read_csv('MovieLens/ratings.dat',
                         sep='::',
                         header=None,
                         names=['uid_old', 'mid_old', 'rating', 'date'],
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, unit='s', origin='unix'),
                         engine='python')

        # DROP MOVIES WITH LESS THAN 5 RATINGS
        s = df.groupby(['mid_old']).size()
        low_n_ratings = s[s < min_ratings].reset_index().mid_old.tolist()
        df = df[~df.mid_old.isin(low_n_ratings)]
        # RE-INDEX USERS AND MOVIES
        user_id = df[['uid_old']].drop_duplicates().reindex()
        user_id['uid'] = np.arange(len(user_id))
        df = pd.merge(df, user_id, on=['uid_old'], how='left')

        item_id = df[['mid_old']].drop_duplicates()
        item_id['mid'] = np.arange(len(item_id))
        return pd.merge(df, item_id, on=['mid_old'], how='left')

    def features(self):
        df = pd.read_csv('MovieLens/users.dat',
                         sep='::',
                         header=None,
                         names=['uid', 'gender', 'age', 'job', 'zip'],
                         engine='python')
        df.drop(columns=['uid'], inplace=True)
        df.index.rename('uid', inplace=True)
        df.gender = pd.get_dummies(df.gender, drop_first=True)  # 0:F, 1:M
        df.reset_index(inplace=True)
        drop = [0, 10, 13, 19]

        clean = df[~df['job'].isin(drop)]

        clean['rating'] = 1

        assert (clean.uid.nunique() == self.num_users)
        self.num_jobs = clean.job.nunique()

        item_id = clean[['job']].drop_duplicates()
        item_id['njob'] = np.arange(self.num_jobs)
        clean = pd.merge(clean, item_id, on=['job'], how='left')
        clean.job = clean.njob
        return clean


class FairnessData(Dataset):
    def __init__(self, num_negatives: int = 4):
        self.df = self.features()
        self.num_users = self.df.uid.nunique()
        self.num_jobs = self.df.mid.nunique()
        self.jobs = set(self.df.job.unique())

        self.train, self.test = self._train_test_split()

        self.training_data = self.add_negatives(self.train, n_samples=num_negatives)

    def __len__(self):
        return self.train.shape[0]  # Length of the data to train on

    def __getitem__(self, index) -> T_co:
        user = LongTensor(self.training_data.uid.iloc[index])
        job = LongTensor(self.training_data.job.iloc[index])
        # protected attribute
        gender = LongTensor(self.training_data.gender.iloc[index])
        rating = LongTensor(self.training_data.rating.iloc[index])
        return user, job, gender, rating

    def train_test_split(self, train_ratio: float = .7):
        msk = np.random.rand(len(self.df)) < train_ratio

        train = self.df[msk]
        test = self.df[~msk]
        return train, test

    def add_negatives(self, df: pd.DataFrame, item: str = 'mid', items=None, n_samples: int = 4):
        if items is None:
            items = set(self.df[item].unique())

        combine = df.groupby('uid')[item].apply(set).reset_index()
        combine['negatives'] = combine[item].apply(lambda x: sample(list(items - x), n_samples))

        s = combine.apply(lambda x: pd.Series(x.negatives, dtype=np.int16), axis=1).stack().reset_index()
        s.rename(columns={'level_0': 'uid', 0: item}, inplace=True)
        s.drop(['level_1'], axis=1, inplace=True)
        s['rating'] = np.int8(0)
        s.uid = s.uid.astype(np.int16)

        complete = pd.concat([df, s]).sort_values(by=['uid', item])
        return complete.reset_index(drop=True)

    @staticmethod
    def load_ratings(min_ratings=5):
        df = pd.read_csv('MovieLens/ratings.dat',
                         sep='::',
                         header=None,
                         names=['uid_old', 'mid_old', 'rating', 'date'],
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, unit='s', origin='unix'),
                         engine='python')

        # DROP MOVIES WITH LESS THAN 5 RATINGS
        s = df.groupby(['mid_old']).size()
        low_n_ratings = s[s < min_ratings].reset_index().mid_old.tolist()
        df = df[~df.mid_old.isin(low_n_ratings)]
        # RE-INDEX USERS AND MOVIES
        user_id = df[['uid_old']].drop_duplicates().reindex()
        user_id['uid'] = np.arange(len(user_id))
        df = pd.merge(df, user_id, on=['uid_old'], how='left')

        item_id = df[['mid_old']].drop_duplicates()
        item_id['mid'] = np.arange(len(item_id))
        return pd.merge(df, item_id, on=['mid_old'], how='left')

    def features(self):
        df = pd.read_csv('MovieLens/users.dat',
                         sep='::',
                         header=None,
                         names=['uid', 'gender', 'age', 'job', 'zip'],
                         engine='python')
        df.drop(columns=['uid'], inplace=True)
        df.index.rename('uid', inplace=True)
        df.gender = pd.get_dummies(df.gender, drop_first=True)  # 0:F, 1:M
        df.reset_index(inplace=True)
        drop = [0, 10, 13, 19]

        clean = df[~df['job'].isin(drop)]

        clean['rating'] = 1

        assert (clean.uid.nunique() == self.num_users)
        self.num_jobs = clean.job.nunique()

        item_id = clean[['job']].drop_duplicates()
        item_id['njob'] = np.arange(self.num_jobs)
        clean = pd.merge(clean, item_id, on=['job'], how='left')
        clean.job = clean.njob
        return clean
