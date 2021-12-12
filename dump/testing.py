from torch import LongTensor
import torch
import numpy as np
import pandas as pd
from time import time
from models import NCF3
from torch.utils.data import DataLoader
from evaluators import eval_model, evaluate_model
from torch.utils.data.dataset import T_co
import random
from numpy.random import choice

num_epochs = 25
batch_size = 2048
learning_rate = .001

emb_size = 128  # LATENT DIM

hidden_layers = 4  # np.array([emb_size, 64, 32, 16])
output_size = 1

random_samples = 100
num_negatives = 5
top_k = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# movies grouped by user
def all_movies_by_usr(train, test):
    movies_all = zip(train.groupby('uid'), test.groupby('uid'))
    return [(set(tr[1].mid.tolist()), set(te[1].mid.tolist())) if tr[1].uid.iloc[0] == te[1].uid.iloc[0] else '' for
            tr, te in movies_all]


# movies in train and movies in test for each is user
def check_for_leaks(train, test):
    # x = 0
    leaks = 0
    u = choice(range(test.uid.nunique()))
    for i, (tr, te) in enumerate(all_movies_by_usr(train, test)):
        if i == u:
            # tr.add(25)  # add test movie to train
            print(f"User: {u} watched these movies:")
            print(f"movies in train {sorted(list(tr)[:5])}")
            print(f"\tmovies in test {sorted(list(te)[:5])}")
            if tr.intersection(te):
                print(f'user: {i} has {tr.intersection(te)} in both train/test')
                leaks += 1
    if leaks > 1:
        print(f"*** {leaks} Leaks ***")
    elif leaks == 1:
        print(f"*** {leaks} Leak ***")
    else:
        print('*** 0 Leaks Found ***')


class LoadData(torch.utils.data.Dataset):
    def __init__(self):
        t1 = time()
        self.ratings = self._load_ratings()
        t2 = time()
        self.user_features = self._load_features()
        t3 = time()
        self.fd = pd.merge(self.ratings, self.user_features, on=['uid', 'uid'], how='right')
        t4 = time()
        self.fd = self._reset_index(self.fd, ['uid', 'mid', 'job'])
        self.train, self.test = self._train_test_split()
        movies = set(self.fd.mid.unique())
        self.tr = self.train  # self.add_negatives(self.train, items=movies, n_samples=5)
        # self.te = self.add_negatives(self.test, items=movies, tr=self.train, n_samples=100)
        self.testing_tensors = self.parse_testing(self.test)

        t6 = time()
        print('ratings: {}s'.format(int(t2 - t1)))
        print('features: {}s'.format(int(t3 - t2)))
        print('negatives: {}s'.format(int(t6 - t4)))
        print('total time: {}s'.format(int(t6 - t1)))

    def __len__(self):
        return self.tr.mid.shape[0]

    def __getitem__(self, index) -> T_co:
        user = LongTensor([self.tr.uid.iloc[index]])
        movie = LongTensor([self.tr.mid.iloc[index]])
        # output = LongTensor([self.tr.rating.iloc[index]])
        return user, movie  # , output

    def add_negatives(self, df: pd.DataFrame, item: str = 'mid', items=None, tr=None, n_samples: int = 5):
        if items is None:
            items = set(self.ratings[item].unique())

        def user_movies(uid, mid):
            train_movies = set()
            m_count = len(mid)
            if tr is not None:
                train_movies = set(tr[tr.uid == uid].mid.unique())
                if not train_movies:
                    print('no movies in train found', uid)
                return random.sample(list(items - train_movies - mid), n_samples)
            return random.sample(list(items - train_movies - mid), min(len(items - mid), n_samples * m_count))
            # return sample(list(items - train_movies - mid), m_count * n_samples)

        df['rating'] = np.int8(1)
        combine = df.groupby('uid')[item].apply(set).reset_index()
        combine['negatives'] = combine.apply(lambda x: user_movies(x.uid, x.mid), axis=1)

        s = combine.apply(lambda x: pd.Series(x.negatives, dtype=np.int16), axis=1).stack().reset_index()
        s.rename(columns={'level_0': 'uid', 0: item}, inplace=True)
        s.drop(['level_1'], axis=1, inplace=True)
        s['rating'] = np.int8(0)
        s.uid = s.uid.astype(np.int16)
        complete = pd.concat([df, s]).sort_values(by=['uid'])
        return complete.reset_index(drop=True)

    def _train_test_split(self):
        self.fd['latest'] = self.fd.groupby(['uid'])['date'].rank(method='first', ascending=False)
        test_bool = self.fd.latest <= 1
        test = self.fd[test_bool]
        train = self.fd[~test_bool]
        return (
            train[['uid', 'mid']], test[['uid', 'mid']]
        )

    @staticmethod
    def _reset_index(df, cols):
        for col in cols:
            old_col = col + '_old'
            df.rename(columns={col: old_col}, inplace=True)
            user_id = df[[old_col]].drop_duplicates().reindex()
            user_id[col] = np.arange(len(user_id))
            df = pd.merge(df, user_id, on=[old_col], how='left')
        return df

    @staticmethod
    def _load_ratings(min_ratings=5):
        df = pd.read_csv('../MovieLens/ratings.dat',
                         sep='::',
                         header=None,
                         names=['uid', 'mid', 'rating', 'date'],
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, unit='s', origin='unix'),
                         engine='python')

        # DROP MOVIES WITH LESS THAN 5 RATINGS
        s = df.groupby(['mid']).size()
        low_n_ratings = s[s < min_ratings].reset_index().mid.tolist()
        return df[~df.mid.isin(low_n_ratings)]

    @staticmethod
    def _load_features():
        df = pd.read_csv('../MovieLens/users.dat',
                         sep='::',
                         header=None,
                         names=['uid', 'gender', 'age', 'job', 'zip'],
                         engine='python')
        df.gender = pd.get_dummies(df.gender, drop_first=True)  # 0:F, 1:M
        drop = [0, 10, 13, 19]

        clean = df[~df['job'].isin(drop)]
        return clean.drop(columns=['zip'])

    @staticmethod
    def parse_testing(df):
        test = df.sort_values(by=['uid', 'rating'], ascending=False)
        users, movies, outputs = [], [], []
        for _, u in test.groupby('uid'):
            users.append(LongTensor(u.uid.to_numpy()))
            movies.append(LongTensor(u.mid.to_numpy()))
            outputs.append(LongTensor(u.rating.to_numpy()))
        return users, movies, outputs


def get_instances_with_random_neg_samples(train, num_items, num_negatives, device):
    u, m = train

    df = pd.DataFrame({'uid': u.squeeze(1).numpy(), 'mid': m.squeeze(1).numpy()})

    n = df.shape[0]
    user_input = np.zeros((n + n * num_negatives))
    item_input = np.zeros((n + n * num_negatives))
    labels = np.zeros((n + n * num_negatives))

    neg_samples = choice(num_items, size=(10 * n * num_negatives,))
    neg_counter = 0
    i = 0
    for n in range(n):
        # positive instance
        user_input[i] = df.uid[n]
        item_input[i] = df.mid[n]
        labels[i] = 1
        i += 1
        # negative instances
        checkList = list(df.mid[df.uid == df.uid[n]])
        for t in range(num_negatives):
            j = neg_samples[neg_counter]
            while j in checkList:
                neg_counter += 1
                j = neg_samples[neg_counter]
            user_input[i] = df.uid[n]
            item_input[i] = j
            labels[i] = 0
            i += 1
            neg_counter += 1
    return torch.LongTensor(user_input).to(device), torch.LongTensor(item_input).to(device), torch.FloatTensor(
        labels).to(device)

def train(model, data, n_users, n_items):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    loss_func = torch.nn.BCELoss()  # (weight=w, reduction="mean")

    it_per_epoch = len(data) / batch_size

    loss = 0
    for i in range(num_epochs):
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)
        model.train()
        j = 0
        t1 = time()
        for batch in dataloader:
            u, m, r = get_instances_with_random_neg_samples(
                batch,
                n_items,
                num_negatives,
                device
            )
            y_hat = ncf(u, m)
            # y_hat = ncf(u.squeeze(1), m.squeeze(1))

            loss = loss_func(y_hat, r.unsqueeze(1))
            # loss = loss_func(y_hat, r.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % int(1 + it_per_epoch / 10) == 0:
                print(f"\rEpoch {i + 1}: {round(100 * j / it_per_epoch)}%", end='')
            j += 1

        t2 = time()
        print("\nEpoch time:", round(t2 - t1), "seconds")
        print("Loss:", loss.cpu().detach().numpy().round(3))
        model.eval()
        hr, ndcg = eval_model(model, data, n_users, device)

        print(f"HR@{top_k}: {round(hr, 2)}  NDCG@{top_k}: {round(ndcg, 2)}\n")
        avg_HR_preTrain, avg_NDCG_preTrain = evaluate_model(
            ncf,
            data.test[['uid', 'mid']].values,
            top_k=10,
            random_samples=100,
            num_items=data.tr.mid.nunique(),
            device=device
        )
        print(avg_HR_preTrain)
    print("Done")


if __name__ == '__main__':
    # %% --- PARSING DATA ---
    print("Processing data...")
    data = LoadData()
    # check_for_leaks(data.tr, data.te)
    print("Done")
    num_users = data.tr.uid.nunique()
    num_items = data.tr.mid.nunique()
    # %% --- LOAD MODEL AND TRAIN ---
    ncf = NCF3(
        num_users,
        num_items,
        emb_size,
        num_layers=4,
        dropout=1
    ).to(device)

    train(ncf, data, num_users, num_items)

    # %% --- SAVE MODEL ---
    # torch.save(ncf.state_dict(), "saved_models/NCF")
    # print('Model Saved')
