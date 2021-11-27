import pandas as pd
import numpy as np
from random import sample
# import tensorflow as tf

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from torch import LongTensor

class TargetData(Dataset):
    def __init__(self, num_negatives=4):
        self.num_jobs = 0
        self.df = self.load_ratings()
        self.num_users = self.df.uid.nunique()
        self.num_movies = self.df.mid.nunique()
        self.users = set(self.df.uid.unique())
        self.movies = set(self.df.mid.unique())

        self.train, self.test = self._train_test_split()

        self.training_data = self.add_negatives(self.train, n_samples=num_negatives)
        self.testing_data = self.add_negatives(self.test, items=self.movies, n_samples=100)
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

        self.train, self.test = self._train_test_split(training_ratio)

        self.tr = self.add_negatives(
            self.train,
            item='job',
            items=self.jobs,
            n_samples=num_negatives)

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

    # def __getitem__(self, index) -> T_co:
    #     user = LongTensor(self.training_data.uid.iloc[index])
    #     job = LongTensor(self.training_data.job.iloc[index])
    #     # protected attribute
    #     gender = LongTensor(self.training_data.gender.iloc[index])
    #     age = LongTensor(self.training_data.age.iloc[index])
    #     rating = LongTensor(self.training_data.rating.iloc[index])
    #     return user, job, gender, age, rating

    # def __call__(self, test_data):
    #     return self.parse_testing(self.add_negatives(test_data, items=self.movies, n_samples=100))

    # def perturb_input(self):
    #     # HOT ENCODING (CATEGORICAL)
    #     func1, func2 = self.obfuscation_functions()
    #
    #     jobs_train = pd.get_dummies(self.train.job, drop_first=True)
    #     jobs_train.apply(func2, inplace=True)
    #
    #     genders_train = pd.get_dummies(self.train.gender, drop_first=True)
    #     genders_train.apply(func2, inplace=True)
    #
    #     # (CONTINUOUS)
    #     ages_train = 2 * ((self.train.age - self.train.age.min()) /
    #                       (self.train.age.max() - self.train.age.min())) - 1
    #     ages_train.apply(func1, inplace=True)
    #     return jobs_train, genders_train, ages_train

    # @staticmethod
    # def obfuscation_functions(eps_hat: int = 4):
    #     n_features = 3  # d
    #
    #     delta = n_features + n_features ** 2 / 4  # global sensitivity
    #     slack = np.random.uniform(0, 1, 1)
    #     # slack = np.max(1, n_features, np.int8(local_epsilon / 2.5))
    #
    #     C = (np.exp(eps_hat / 2) + 1) / (np.exp(eps_hat / 2) - 1)
    #
    #     lx = lambda x: ((C + 1) / 2) * x - ((C - 1) / 2)
    #     pix = lambda x: lx(x) + C - 1
    #
    #     dfrac = np.exp(eps_hat / 2) / np.exp(eps_hat / 2) + 1
    #
    #     func1 = lambda x: np.random.uniform(lx(x), pix(x), 1) if slack < dfrac else np.random.choice(
    #         [np.random.uniform(-C, lx(x), 1), np.random.uniform(pix(x), C, 1)], 1)
    #
    #     func2 = lambda x: np.float32(.5) if x == 1 else 1 / (np.exp(eps_hat / slack) + 1)
    #     return func1, func2

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
        msk = np.random.rand(len(self.df)) < train_ratio

        train = self.df[msk]
        test = self.df[~msk]
        return train, test

    def _features(self):
        df = pd.read_csv('MovieLens/users.dat',
                         sep='::',
                         header=None,
                         names=['uid', 'gender', 'age', 'ojob', 'zip'],
                         engine='python')
        df.drop(columns=['uid'], inplace=True)
        df.index.rename('uid', inplace=True)
        df.gender = pd.get_dummies(df.gender, drop_first=True)  # 0:F, 1:M
        df.reset_index(inplace=True)
        drop = [0, 10, 13, 19]

        clean = df[~df['ojob'].isin(drop)]

        clean['rating'] = 1
        clean['uid'] = clean.uid - 1

        self.num_jobs = clean.ojob.nunique()

        item_id = clean[['ojob']].drop_duplicates()
        item_id['job'] = np.arange(self.num_jobs)
        clean = pd.merge(clean, item_id, on=['ojob'], how='left')
        # clean.job = clean.njob
        clean.drop(columns=['zip'], inplace=True)
        return clean


class DataGenerator(AttributeData):
    def __init__(self):
        super().__init__()
        # self.features = AttributeData()
        self.targets = TargetData()

        self.complete = self.pd.merge(self.features.df, self.targets.training_data, on=['uid', 'uid'], how='left')
        self.train, self.test = self._train_test_split()
        self.jobs, self.genders, self.ages = self.perturb_input()

    def __len__(self):
        return self.ages_train.shape[0]  # Length of the data to train on

    def _train_test_split(self, **kwargs):
        self.complete['latest'] = self.complete.groupby(['uid'])['date'].rank(method='first', ascending=False)
        test_bool = self.df.latest <= 1
        test = self.df[test_bool]
        train = self.df[~test_bool]
        return (train[['uid', 'mid', 'age', 'gender', 'job', 'rating']],
                test[['uid', 'mid', 'age', 'gender', 'job', 'rating']]
                )

    @staticmethod
    def parse_testing(df):
        test = df.sort_values(by=['uid', 'rating'], ascending=False)
        users, movies, outputs = [], [], []
        for _, u in test.groupby('uid'):
            users.append(LongTensor([u.uid.values]))
            movies.append(LongTensor([u.mid.values]))
            outputs.append(LongTensor([u.rating.values]))
        return users, movies, outputs
