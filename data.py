import pandas as pd
import numpy as np
from random import sample
# import tensorflow as tf

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from torch import LongTensor


class MovieTrainData(Dataset):
    def __init__(self, train):
        self.train = train

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, index) -> T_co:
        user = LongTensor([self.train.uid.iloc[index]])
        movie = LongTensor([self.train.mid.iloc[index]])
        output = LongTensor([self.train.rating.iloc[index]])
        return user, movie, output


class TargetData(Dataset):
    def __init__(self, num_negatives=4):
        f = AttributeData()

        self.num_jobs = 0
        self.df = self.load_ratings()
        # self.num_users = self.df.uid.nunique()
        # self.num_movies = self.df.mid.nunique()
        # self.users = set(self.df.uid.unique())
        # self.movies = set(self.df.mid.unique())

        self.train, self.test = self._train_test_split()

        self.complete_train = pd.merge(f.df, self.train, on=['uid', 'uid'], how='left')
        self.num_users = self.complete_train.uid.nunique()
        self.num_movies = self.complete_train.mid.nunique()

        self.complete_train['uid_old'] = self.complete_train.uid

        user_id = self.complete_train[['uid_old']].drop_duplicates()
        user_id['uid'] = np.arange(self.num_users)
        self.complete_train = pd.merge(self.complete_train, user_id, on=['uid_old'], how='left')
        self.complete_train.rename(columns={'uid_y': 'uid', 'rating_y': 'rating'}, inplace=True)
        self.complete_train.drop(columns=['uid_x', 'ojob'], inplace=True)
        self.users = set(self.complete_train.uid.unique())
        self.movies = set(self.complete_train.mid.unique())

        self.training_data = self.add_negatives(self.complete_train, n_samples=num_negatives)

        self.complete_test = pd.merge(f.df, self.test, on=['uid', 'uid'], how='left')
        self.complete_test['uid_old'] = self.complete_test.uid

        user_id = self.complete_test[['uid_old']].drop_duplicates()
        user_id['uid'] = np.arange(self.num_users)
        self.complete_test = pd.merge(self.complete_test, user_id, on=['uid_old'], how='left')
        self.complete_test.rename(columns={'uid_y': 'uid', 'rating_y': 'rating'}, inplace=True)
        self.complete_test.drop(columns=['uid_x', 'ojob'], inplace=True)

        self.testing_data = self.add_negatives(self.complete_test, items=self.movies, n_samples=100)
        self.testing_tensors = self.parse_testing(self.testing_data)

    def __len__(self):
        return self.training_data.shape[0]  # Length of the data to train on

    def __getitem__(self, index) -> T_co:
        user = LongTensor([self.training_data.uid.iloc[index]])
        movie = LongTensor([self.training_data.mid.iloc[index]])
        output = LongTensor([self.training_data.rating.iloc[index]])
        return user, movie, output

    def __call__(self, test_data):
        return self.parse_testing(self.add_negatives(test_data, items=self.movies, n_samples=100))

    @staticmethod
    def load_ratings(min_ratings=5):
        df = pd.read_csv('MovieLens/ratings.dat',
                         sep='::',
                         header=None,
                         names=['uid', 'mid_old', 'rating', 'date'],
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, unit='s', origin='unix'),
                         engine='python')

        # DROP MOVIES WITH LESS THAN 5 RATINGS
        s = df.groupby(['mid_old']).size()
        low_n_ratings = s[s < min_ratings].reset_index().mid_old.tolist()
        df = df[~df.mid_old.isin(low_n_ratings)]

        item_id = df[['mid_old']].drop_duplicates()
        item_id['mid'] = np.arange(len(item_id))
        return pd.merge(df, item_id, on=['mid_old'], how='left')

    @staticmethod
    def parse_testing(df):
        test = df.sort_values(by=['uid', 'rating'], ascending=False)
        users, movies, outputs = [], [], []
        for _, u in test.groupby('uid'):
            users.append(LongTensor(u.uid.to_numpy()))
            # users.append(LongTensor([u.uid.values]))
            movies.append(LongTensor(u.mid.to_numpy()))
            # movies.append(LongTensor([u.mid.values]))
            outputs.append(LongTensor(u.rating.to_numpy()))
            # outputs.append(LongTensor([u.rating.values]))
        return users, movies, outputs

    def _train_test_split(self):
        self.df.rating = np.int8(1)
        self.df['latest'] = self.df.groupby(['uid'])['date'].rank(method='first', ascending=False)
        test_bool = self.df.latest <= 1
        test = self.df[test_bool]
        train = self.df[~test_bool]
        return (train[['uid', 'mid', 'rating']],
                test[['uid', 'mid', 'rating']]
                )

    def add_negatives(self, df: pd.DataFrame, item: str = 'mid', items=None, n_samples: int = 4):
        if items is None:
            items = set(self.train[item].unique())

        combine = df.groupby('uid')[item].apply(set).reset_index()
        combine['negatives'] = combine[item].apply(lambda x: sample(list(items - x), n_samples))

        s = combine.apply(lambda x: pd.Series(x.negatives, dtype=np.int16), axis=1).stack().reset_index()
        s.rename(columns={'level_0': 'uid', 0: item}, inplace=True)
        s.drop(['level_1'], axis=1, inplace=True)
        s['rating'] = np.int8(0)
        s.uid = s.uid.astype(np.int16)

        complete = pd.concat([df, s]).sort_values(by=['uid', item])
        return complete.reset_index(drop=True)


class AttributeData(Dataset):
    def __init__(self, num_negatives: int = 4, training_ratio: float = .8):
        self.df = self._features()
        self.num_users = self.df.uid.nunique()
        self.num_jobs = self.df.job.nunique()
        self.jobs = set(self.df.job.unique())
        self.all_data = self.df.copy()
        self.all_data['uid_old'] = self.all_data.uid
        user_id = self.all_data[['uid_old']].drop_duplicates()
        user_id['uid'] = np.arange(self.num_users)
        self.all_data = pd.merge(self.all_data, user_id, on=['uid_old'], how='left')
        self.all_data.drop(columns=['uid_x', 'index'], inplace=True)
        self.all_data.rename(columns={'uid_y': 'uid'}, inplace=True)
        self.train, self.test = self._train_test_split(training_ratio)
        self.tr = self.add_negatives(
            self.train,
            item='job',
            items=self.jobs,
            n_samples=num_negatives
        )

        self.tr['age'] = self.tr.groupby('uid')['age'].transform('first')
        self.tr['gender'] = self.tr.groupby('uid')['gender'].transform('first')
        self.tr = self.tr[self.tr['gender'].notna()]

        self.tr.gender = self.tr.gender.astype(np.int8)
        self.tr.age = self.tr.age.astype(np.int16)
        # self.jobs, self.genders, self.ages = self.perturb_input()

    def __len__(self):
        return self.tr.shape[0]  # Length of the data to train on

    def __getitem__(self, index) -> T_co:
        user = LongTensor([self.tr.uid.iloc[index]])
        job = LongTensor([self.tr.job.iloc[index]])
        gender = LongTensor([self.tr.gender.iloc[index]])
        # age = LongTensor(self.training_data.age.iloc[index])
        rating = LongTensor([self.tr.rating.iloc[index]])
        return user, job, gender, rating

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

        # complete = pd.merge([df, s], how='left', on='uid').sort_values(by=['uid', item])
        complete = pd.concat([df, s]).sort_values(by=['uid', item])
        return complete.reset_index(drop=True)

    def _train_test_split(self, train_ratio: float):
        msk = np.random.rand(len(self.all_data)) < train_ratio

        train = self.all_data[msk]
        test = self.all_data[~msk]
        return train, test

    def _features(self):
        df = pd.read_csv('MovieLens/users.dat',
                         sep='::',
                         header=None,
                         names=['uid', 'gender', 'age', 'ojob', 'zip'],
                         engine='python')
        # df.drop(columns=['uid'], inplace=True)
        # df.index.rename('uid', inplace=True)
        df.gender = pd.get_dummies(df.gender, drop_first=True)  # 0:F, 1:M
        drop = [0, 10, 13, 19]

        clean = df[~df['ojob'].isin(drop)].copy()

        clean['rating'] = 1
        # clean['uid'] = clean.uid.copy() - 1

        self.num_jobs = clean.ojob.nunique()

        item_id = clean[['ojob']].drop_duplicates()
        item_id['job'] = np.arange(self.num_jobs)
        clean = pd.merge(clean, item_id, on=['ojob'], how='left')
        # clean.job = clean.njob
        clean.drop(columns=['zip'], inplace=True)
        return clean.reset_index()

