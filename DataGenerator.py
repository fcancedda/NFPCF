import pandas as pd
import numpy as np
from random import sample


class DataGenerator:
    def __init__(self):
        self.df = self.load_ratings()
        self.num_users = self.df.uid.nunique()
        self.num_movies = self.df.mid.nunique()

        self.users = set(self.df.uid.unique())
        self.movies = set(self.df.mid.unique())

        self.train, self.test = self._train_test_split()
        self.train = self.add_negatives(n_samples=4)
        self.test_pos = self.load_test_data()
        self.test_neg = self.load_negatives()



    def _train_test_split(self):
        self.df.rating = np.int8(1)
        self.df['latest'] = self.df.groupby(['uid'])['time'].rank(method='first', ascending=False)
        test = self.df[self.df.latest == 1]
        train = self.df[self.df.latest > 1]
        return (train[['uid', 'mid', 'rating']].set_index('uid'),
                test[['uid', 'mid', 'rating']].set_index('uid'))

    def add_negatives(self, df, n_samples=4):
        combine = df.groupby('uid')['mid'].apply(set).reset_index()
        combine['negatives'] = combine.mid.apply(lambda m: sample(list(self.movies - m), n_samples))

        s = combine.apply(lambda m: pd.Series(m.negatives, dtype=np.int16), axis=1).stack().reset_index()
        s.rename(columns={'level_0': 'uid', 0: 'mid'}, inplace=True)
        s.drop(['level_1'], axis=1, inplace=True)
        s['rating'] = np.int8(0)
        s.uid = s.uid.astype(np.int16)

        complete = pd.concat([df, s]).sort_values(by=['uid', 'mid'])
        return complete.reset_index(drop=True)

    @staticmethod
    def load_ratings():
        return pd.read_csv('MovieLens/ratings.dat',
                           sep='::',
                           header=None,
                           names=['uid', 'mid', 'rating', 'date'],
                           parse_dates=['date'],
                           date_parser=lambda x: pd.to_datetime(x, unit='s', origin='unix'),
                           engine='python')
    @staticmethod
    def get_features_as_list(df):
        return df.apply(lambda x: [x.uid, x.mid], axis=1).tolist()

    def get_test_negatives(self):
        df = pd.read_csv('Data/ml-1m.test.negative',
                         sep='\t',
                         header=None,
                         )
        df.drop(columns=[0], inplace=True, axis=1)
        df.reset_index(drop=True, inplace=True)
        df.head()
        return df.values.tolist()
