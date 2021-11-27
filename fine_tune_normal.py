""" FINE TUNING MODEL  WITHOUT FAIRNESS OR DIFFERENTIAL PRIVACY """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import NCF3
from data import AttributeData, TargetData
# from fairness_measures import Measures
from evaluators import evaluate_model
from fairness_measures import Measures
# CONSTANTS
emb_size = 128
# hidden_layers = np.array([emb_size, 64, 32, 16])
num_layers = 4
output_size = 1

num_epochs = 10
batch_size = 128

num_negatives = 5

random_samples = 15
top_k = 10

learning_rate = .01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fairness_thres = torch.tensor(0.1).to(device)
epsilonBase = torch.tensor(0.0).to(device)

# LOAD DATA AND FAIRNESS FUNCTIONS
td = TargetData()
data = AttributeData()
# m = Measures()


def load_model():
    # LOAD PRE-TRAINED MODEL
    ncf = NCF3(td.num_users, td.num_movies, emb_size, num_layers, output_size).to(device)
    # ncf = NCF(data.num_users, data.num_jobs, emb_size, hidden_layers, output_size).to(device)
    ncf.load_state_dict(torch.load("saved_models/NCF2"))

    # FETCH NUMBER OF UNIQUE CAREERS
    n_careers = data.num_jobs
    # CHANGE EMBEDDING SIZE TO FIT SENSITIVE INFO
    ncf.embed_item_GMF = nn.Embedding(n_careers, emb_size).to(device)
    ncf.embed_item_MLP = nn.Embedding(n_careers, emb_size * (2 ** (num_layers - 1))).to(device)
    ncf.embed_item_GMF.weight.requires_grad = False
    ncf.embed_item_MLP.weight.requires_grad = False
    return ncf


def train_normal(model):
    # BINARY CROSS-ENTROPY LOSS + ADAM OPTIMIZER
    loss = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
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
            y_hat = model(users.squeeze(1), jobs.squeeze(1))

            final_loss = loss(y_hat, ratings.float())

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            if j % int(1 + it_per_epoch / 10) == 0:
                print(f"\r Epoch {i + 1}, Progress: {round(100 * j / it_per_epoch)}%", end='', flush=True)

            j += 1
        ht, ndcg = evaluate_model(ncf, data.test[['uid', 'job']].values, top_k, random_samples, data.num_jobs, device)

        print(f'\nHit Ratio: {round(ht[-1], 2)}  NDCG: {round(ndcg[-1], 2)}   LOSS1: {final_loss}')


if __name__ == '__main__':
    ncf = load_model()
    hr, ndcg = evaluate_model(
        ncf,
        data.test[['uid', 'job']].values,
        top_k,
        random_samples,
        data.num_jobs,
        device
    )
    print(f'\nHit Ratio: {round(hr[-1], 2)}  NDCG: {round(ndcg[-1], 2)}')

    train_normal(ncf)

    m = Measures()

    all_genders = torch.LongTensor(data.train['gender'].values).to(device)
    m.fairness(ncf, data.test, all_genders.cpu(), data.num_jobs, device)

    torch.save(ncf.state_dict(), "saved_models/tunedNCF")

