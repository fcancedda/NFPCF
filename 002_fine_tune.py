import numpy as np

import torch
from torch.utils.data import DataLoader

from data import load_data, add_false, set_seed
from models import NCF
from evaluators import eval_results

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    n = len(dataloader)
    for i in range(num_epochs):
        pct = 0
        for batch in dataloader:
            model.train()

            usr, itm, rtn = add_false(batch, n_false=num_negatives, n_items=n_careers, device=device)
            y_hat = model(usr, itm)

            loss = criterion(y_hat, rtn.unsqueeze(1))

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
            device=device,
            fairness=True,
            k=top_k
        )


# %% MAIN
if __name__ == "__main__":
    set_seed(54)
    emb_size = 128
    hidden_layers = np.array([emb_size, 64, 32, 16])
    output_size = 1
    num_epochs = 10
    batch_size = 256
    num_negatives = 5
    random_samples = 15
    top_k = 5
    lr = .001  # LEARNING RATE
    # FETCH NUMBER OF UNIQUE CAREERS
    n_careers = 17  # len(dataset.tensors[1].unique())
    pre = True  # USE PRETRAINED MODEL

    # LOAD PRE-TRAINED MODEL
    if (pre):  # IF PRE IS TRUE
        ncf = NCF(6040, 3416, emb_size, hidden_layers, output_size).to(device)
        ncf.load_state_dict(torch.load("output/preTrained_NCF"))
        eval_results(model=ncf, n_items=3416, mode='RATINGS', note='PRE DE-BIAS', device=device)
    else:
        ncf = NCF(6040, n_careers, emb_size, hidden_layers, output_size).to(device)

    dataset = load_data(pre_train=False, device=device)

    '''UPDATE USER EMBEDDINGS'''
    # replace page items with career items
    ncf.like_emb = torch.nn.Embedding(n_careers, emb_size).to(device)
    ncf.user_emb.weight.requires_grad = False

    eval_results(model=ncf, n_items=n_careers, n_samples=random_samples, mode='CAREERS', k=top_k, note='POST DE-BIAS',
                 device=device)

    fine_tune(ncf, dataset)
    eval_results(model=ncf, n_items=n_careers, n_samples=random_samples, mode='CAREERS', k=top_k, note='Fine-Tuning',
                 print_metrics='unfair', device=device)
    torch.save(ncf.state_dict(), "output/unfair_NCF")  # SAVE MODEL
