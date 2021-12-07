import numpy as np

import torch
from torch.utils.data import DataLoader

from data import load_data, add_false
from models import NCF
from evaluators import eval_results
from fairness_measures import Measures

m = Measures()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% DEBIAS CURRENT USER EMBEDDING
def de_bias_embedding(model, data):
    inputs, targets = data  # ( UID, MOVIE )
    # LOAD USER EMBEDDING AND WEIGH BY GENDER
    user_embeds = model.user_emb.weight.data.cpu().detach().numpy()
    user_embeds = user_embeds.astype('float')

    ''' COMPUTE GENDER EMBEDDING '''
    gender_embed = np.zeros((2, user_embeds.shape[1]))
    num_users_x_group = np.zeros((2, 1))

    for i in range(len(inputs)):
        user, item, gender = inputs[i]
        if gender == 0:
            gender_embed[0] += user_embeds[user]
            num_users_x_group[0] += 1.0
        else:
            gender_embed[1] += user_embeds[user]
            num_users_x_group[1] += 1.0

    ''' VERTICAL BIAS'''
    gender_embed = gender_embed / num_users_x_group
    # vBias = compute_bias_direction(gender_embed)
    vBias = gender_embed[1].reshape((1, -1)) - gender_embed[0].reshape((1, -1))
    vBias = vBias / np.linalg.norm(vBias, axis=1, keepdims=1)

    ''' LINEAR PROJECTION '''
    debiased_user_embeds = user_embeds
    for i in range(len(inputs)):
        user, item, gender = inputs[i]
        debiased_user_embeds[user] = user_embeds[user] - (
            np.inner(user_embeds[user].reshape(1, -1), vBias)[0][0]) * vBias

    print(f'males / female {max(num_users_x_group) / min(num_users_x_group)}')
    return torch.from_numpy(debiased_user_embeds.astype(np.float32)).to(device)


# %% FAIRNESS CALCULATOR
def fairness_measures(model, df_val, num_items, protectedAttributes):
    model.eval()
    user_input = torch.LongTensor(df_val['user_id'].values).to(device)
    item_input = torch.LongTensor(df_val['like_id'].values).to(device)
    y_hat = model(user_input, item_input)

    avg_epsilon = m.compute_edf(
        protected_attributes=protectedAttributes,
        predictions=y_hat,
        n_classes=num_items,
        item_input=item_input,
        device=device
    )
    U_abs = m.compute_absolute_unfairness(protectedAttributes, y_hat, num_items, item_input, device=device)

    avg_epsilon = avg_epsilon.cpu().detach().numpy().reshape((-1,)).item()
    print(f"average differential fairness: {avg_epsilon: .3f}")

    U_abs = U_abs.cpu().detach().numpy().reshape((-1,)).item()
    print(f"absolute unfairness: {U_abs: .3f}")


# %% FINE-TUNE FUNCTION
def fine_tune(model, data, batch_size: int = 256, num_negatives: int = 16, n_jobs: int = 17, epochs: int = 25,
              lr: float = .001):
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0
    )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    all_user_input, all_item_input, all_gender_input = data.tensors[0][:, 0], data.tensors[0][:, 1], data.tensors[0][:,
                                                                                                     2]
    n = len(dataloader)
    for i in range(epochs):
        pct = 0
        for batch in dataloader:
            model.train()

            usr, itm, rtn = add_false(batch, num_negatives, n_jobs, device)
            y_hat = model(usr, itm)

            loss1 = criterion(y_hat, rtn.unsqueeze(1))

            model.eval()
            predicted_probs = model(all_user_input, all_item_input)
            avg_epsilon = m.compute_edf(all_gender_input, predicted_probs, n_jobs, all_item_input, device)
            loss2 = torch.max(torch.tensor(0.0).to(device), (avg_epsilon - epsilonBase))

            loss = loss1 + fairness_thres * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if int(pct / n) % 10:
                print(f'\rEpoch:{i} -- {int(pct / n)}% -- loss: {loss}', end='')
            pct += 100

        eval_results(model=model, n_items=n_jobs, note='Fine-Tuning')


# %% MAIN
if __name__ == "__main__":
    emb_size = 128
    hidden_layers = np.array([emb_size, 64, 32, 16])
    output_size = 1
    num_epochs = 10
    batch_size = 256
    num_negatives = 5
    # random_samples = 15
    top_k = 10
    learning_rate = .001
    # LOAD PRE-TRAINED MODEL
    ncf = NCF(6040, 3952, emb_size, hidden_layers, output_size).to(device)
    ncf.load_state_dict(torch.load("models/preTrained_NCF"))
    # ds = LoadData()
    eval_results(model=ncf, n_items=3952, note='PRE DE-BIAS')

    dataset = load_data(pre_train=False, device=device)

    # FETCH NUMBER OF UNIQUE CAREERS
    n_careers = 17  # len(dataset.tensors[1].unique())
    '''UPDATE USER EMBEDDINGS'''
    fairness_thres = torch.tensor(0.1).to(device)
    epsilonBase = torch.tensor(0.0).to(device)

    # replace page items with career items
    ncf.like_emb = torch.nn.Embedding(n_careers, emb_size).to(device)
    # freeze user embedding
    ncf.user_emb.weight.requires_grad = False
    # replace user embedding of the model with debiased embeddings

    ncf.user_emb.weight.data = de_bias_embedding(ncf, dataset.tensors)

    eval_results(model=ncf, n_items=n_careers, note='POST DE-BIAS')

    fine_tune(ncf, dataset, batch_size=batch_size, n_jobs=n_careers)
