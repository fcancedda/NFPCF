import pandas as pd
import numpy as np
from random import sample
import tensorflow as tf


class DataGenerator:
    def __init__(self):
        self.df = self.load_ratings()
        self.num_users = self.df.uid.nunique()
        self.num_movies = self.df.mid.max() + 1

        self.users = set(self.df.uid.unique())
        self.movies = set(self.df.mid.unique())

        self.train, self.test = self._train_test_split()

    def _train_test_split(self):
        self.df.rating = np.int8(1)
        self.df['latest'] = self.df.groupby(['uid'])['date'].rank(method='first', ascending=False)
        test = self.df[self.df.latest == 1]
        train = self.df[self.df.latest > 1]
        return (train[['uid', 'mid', 'rating']],
                test[['uid', 'mid', 'rating']])

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
    def load_ratings():
        df = pd.read_csv('MovieLens/ratings.dat',
                         sep='::',
                         header=None,
                         names=['uid', 'mid', 'rating', 'date'],
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, unit='s', origin='unix'),
                         engine='python')
        df.uid = df.uid - 1
        df.mid = df.mid - 1
        return df

    @staticmethod
    def get_features(df):
        return [
            df.uid.to_numpy(),
            df.mid.to_numpy()
        ]

    @staticmethod
    def get_dataset(df):
        return tf.data.Dataset.from_tensor_slices(dict(df))

    @staticmethod
    def get_target(df):
        return df.rating.to_numpy()
