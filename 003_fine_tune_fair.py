import numpy as np

import torch
from torch.utils.data import DataLoader

from data import load_data, add_false, set_seed
from models import NCF
from evaluators import eval_results
from fairness_measures import compute_edf

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% DEBIAS CURRENT USER EMBEDDING
def de_bias_embedding(emb, data):
    inputs, targets = data  # ( UID, MOVIE )
    # LOAD USER EMBEDDING AND WEIGH BY GENDER
    user_embeds = emb.weight.data.cpu().detach().numpy()
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


# %% FINE-TUNE FUNCTION
def fine_tune(model, data):
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        # pin_memory=True,
        # num_workers=1
    )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    all_user_input, all_item_input, all_gender_input = (
        data.tensors[0][:, 0],
        data.tensors[0][:, 1],
        data.tensors[0][:, 2]
    )
    n = len(dataloader)
    for i in range(num_epochs):
        pct = 0
        for batch in dataloader:
            model.train()

            usr, itm, rtn = add_false(batch, n_false=num_negatives, n_items=n_careers, device=device)
            y_hat = model(usr, itm)

            loss1 = criterion(y_hat, rtn.unsqueeze(1))

            predicted_probs = model(all_user_input, all_item_input)

            # COMPUTE DIFFERENTIAL FAIRNESS
            avg_epsilon = compute_edf(all_gender_input.cpu(), predicted_probs, n_careers, all_item_input, device)

            loss2 = torch.max(torch.tensor(0.0).to(device), (avg_epsilon - epsilonBase))

            # UPDATE LOSS, PENALIZING UNFAIRNESS
            loss = loss1 + fairness_thres * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if int(pct / n) % 10:
                print(f'\rEpoch:{i} -- {int(pct / n)}% -- loss: {loss}', end='')
            pct += 100

        eval_results(
            model=model,
            n_items=n_careers,
            n_samples=random_samples,
            mode='CAREERS',
            note='Fine-Tuning',
            # print_metrics='NFCF_lmbda_5',
            fairness=True,
            k=top_k,
            device=device
        )


# %% MAIN
if __name__ == "__main__":
    set_seed(54)
    # FETCH NUMBER OF UNIQUE CAREERS
    n_careers = 17  # len(dataset.tensors[1].unique())
    '''UPDATE USER EMBEDDINGS'''
    fairness_thres = torch.tensor(0.1).to(device)
    epsilonBase = torch.tensor(0.0).to(device)

    emb_size = 128
    hidden_layers = np.array([emb_size, 64, 32, 16])
    output_size = 1
    num_epochs = 10
    batch_size = 256
    num_negatives = 5
    random_samples = 15
    top_k = 5
    lr = .001  # LEARNING RATE
    # LOAD PRE-TRAINED MODEL
    ncf = NCF(6040, 3416, emb_size, hidden_layers, output_size).to(device)
    # ncf = NCF3(6040, 3416, emb_size, 4, output_size).to(device)
    ncf.load_state_dict(torch.load("output/preTrained_NCF"))
    eval_results(model=ncf, n_items=3416, mode='RATINGS', note='PRE DE-BIAS', device=device)

    dataset = load_data(pre_train=False, device=device)

    # replace page items with career items
    ncf.like_emb = torch.nn.Embedding(n_careers, emb_size).to(device)
    # ncf.embed_item_GMF = torch.nn.Embedding(n_careers, emb_size).to(device)
    # ncf.embed_item_MLP = torch.nn.Embedding(n_careers, emb_size).to(device)
    # freeze user embedding
    ncf.user_emb.weight.requires_grad = False
    # ncf.embed_user_GMF.weight.requires_grad = False
    # ncf.embed_user_MLP.weight.requires_grad = False
    # replace user embedding of the model with debiased embeddings

    ncf.user_emb.weight.data = de_bias_embedding(ncf.user_emb, dataset.tensors)
    # ncf.embed_user_GMF.weight.data = de_bias_embedding(ncf.embed_user_GMF, dataset.tensors)
    # ncf.embed_user_MLP.weight.data = de_bias_embedding(ncf.embed_user_MLP, dataset.tensors)

    eval_results(
        model=ncf,
        n_items=n_careers,
        n_samples=random_samples,
        mode='CAREERS',
        note='POST DE-BIAS',
        fairness=True,
        k=top_k,
        device=device
    )

    fine_tune(ncf, dataset)
    eval_results(
        model=ncf,
        n_items=n_careers,
        n_samples=random_samples,
        mode='CAREERS',
        note='Fine-Tuning-Fair',
        print_metrics=f'NFCF_lmbda_{str(fairness_thres.cpu().numpy())[-1]}',
        k=top_k,
        device=device
    )
    torch.save(ncf.state_dict(), f"output/NFCF_lmbda_{str(fairness_thres.cpu().numpy())[-1]}")  # SAVE MODEL
