""" FINE TUNING MODEL  WITHOUT FAIRNESS OR DIFFERENTIAL PRIVACY """
import heapq
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from saved_models import NCF
from fairness_measures import Measures
import data

from importlib import reload
reload(data)
from data import AttributeData, TargetData
# CONSTANTS
emb_size = 128
hidden_layers = np.array([emb_size, 64, 32, 16])
output_size = 1

num_epochs = 10
batch_size = 256

num_negatives = 5

random_samples = 15
top_k = 10

learning_rate = .001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fairness_thres = torch.tensor(0.1).to(device)
epsilonBase = torch.tensor(0.0).to(device)

# LOAD DATA AND FAIRNESS FUNCTIONS
data = AttributeData()
m = Measures()


def load_model():
    # LOAD PRE-TRAINED MODEL
    ncf = NCF(data.num_users, data.num_jobs, emb_size, hidden_layers, output_size).to(device)
    ncf.load_state_dict(torch.load("saved_models/preTrained_NCF"))

    # FETCH NUMBER OF UNIQUE CAREERS
    n_careers = data.num_jobs

    # CHANGE EMBEDDING SIZE TO FIT SENSITIVE INFO
    ncf.like_emb = nn.Embedding(n_careers, emb_size).to(device)
    return ncf


def train_normal(train_fraction):
    # REMOVES JOBS BASED ON THRESHOLD + SPLIT DATA
    train, test = data.train_test_split(train_fraction)
    # LOAD TRAINING DATA
    all_users = torch.LongTensor(train['uid'].values).to(device)
    all_items = torch.LongTensor(train['job'].values).to(device)

    # PROTECTED ATTRIBUTE
    all_genders = torch.LongTensor(train['gender'].values).to(device)

    num_batches = np.int64(np.floor(train.shape[0] / batch_size))
    ncf = load_model()
    # BINARY CROSS-ENTROPY LOSS + ADAM OPTIMIZER
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(ncf.parameters(), lr=learning_rate, weight_decay=1e-6)
    final_loss = 0
    for i in range(num_epochs):
        j = 0  # track training progress
        dataloader = DataLoader(data, batch_size=batch_size,
                                shuffle=True, num_workers=0)

        it_per_epoch = len(data) / batch_size

        for batch in dataloader:
            usr, jb, _, rt = batch
            # move tensors to cuda
            users = usr.to(device)
            jobs = jb.to(device)  # career
            # genders = g.to(device)
            ratings = rt.to(device)

            y_hat = ncf(users.squeeze(1), jobs.squeeze(1))

            final_loss = loss(y_hat, ratings.float())

            predicted_probs = ncf(all_users, all_items)
            # avg_epsilon = m.compute_edf(all_genders, predicted_probs, data.num_jobs, all_items, device)

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            if j % int(1 + it_per_epoch / 10) == 0:
                print(f"Progress: {round(100 * j / it_per_epoch)}%")
            j += 1
        ht, ndcg = evaluate_fine_tune(ncf, test, top_k, random_samples)
        print(f'Hit Ratio: {ht}  NDCG: {ndcg}   LOSS1: {final_loss}')


def evaluate_fine_tune(model, df_val, k, random_samples):
    model.eval()
    avg_hr = np.zeros((len(df_val), k))
    avg_ndcg = np.zeros((len(df_val), k))

    for i in range(len(df_val)):
        test_df = data.add_negatives(
            df_val,
            item='job',
            items=data.jobs,
            n_samples=random_samples
        )
        users, items = torch.LongTensor(test_df.uid).to(device), torch.LongTensor(test_df.job).to(device)
        y_hat = model(users, items)

        y_hat = y_hat.cpu().detach().numpy().reshape((-1,))
        items = items.cpu().detach().numpy().reshape((-1,))
        map_item_score = {}
        for j in range(len(y_hat)):
            map_item_score[items[j]] = y_hat[j]
        for k in range(k):
            # Evaluate top rank list
            ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
            gtItem = items[0]
            avg_hr[i, k] = m.get_hit_ratio(ranklist, gtItem)
            avg_ndcg[i, k] = m.get_ndcg(ranklist, gtItem)
        avg_hr = np.mean(avg_hr, axis=0)
        avg_ndcg = np.mean(avg_ndcg, axis=0)
        return avg_hr, avg_ndcg


def run():
    train_ratio = 0.7
    train_normal(train_ratio)


if __name__ == '__main__':
    run()
