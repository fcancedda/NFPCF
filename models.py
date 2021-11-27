
import torch
from torch import nn
import torch.nn.functional as F


# --- Class definition of the MLP model ---
class MLP(nn.Module):
    ''' Constructs a Multi-Layer Perceptron model'''

    def __init__(self, num_users, num_items, embeddings):
        torch.manual_seed(0)
        super().__init__()

        # user and item embedding layers
        self.user_embedding = nn.Embedding(num_users, embeddings).cuda()
        self.item_embedding = nn.Embedding(num_items, embeddings).cuda()

        # MLP layers
        self.l1 = nn.Linear(embeddings * 2, 64).cuda()
        self.l2 = nn.Linear(64, 32).cuda()
        self.l3 = nn.Linear(32, 16).cuda()
        self.l4 = nn.Linear(16, 1, bias=False).cuda()

    def forward(self, user, item):
        # map to embeddings
        embedding1 = self.user_embedding(user).squeeze(1)
        embedding2 = self.item_embedding(item).squeeze(1)

        # Concatenation of the embedding layers
        out = torch.cat((embedding1, embedding2), 1)

        # feed through the MLP layers
        out = F.relu(self.l1(out))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))

        # output between 0 and 1
        out = torch.sigmoid(self.l4(out))
        return out


class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
        self.out_act = nn.Sigmoid()

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        out = self.out_act((U * V).sum(1) + b_u + b_v)
        return out


# --- Class definition of the GMF model ---
class GMF(nn.Module):
    ''' Constructs a Generalized Matrix Factorization model '''

    def __init__(self, num_users, num_items, embeddings):
        torch.manual_seed(0)
        super().__init__()

        # user and item embedding layers
        self.user_embedding = nn.Embedding(num_users, embeddings).cuda()
        self.item_embedding = nn.Embedding(num_items, embeddings).cuda()

    def forward(self, user, item):
        # map to embeddings
        embedding1 = self.user_embedding(user).squeeze(1)
        embedding2 = self.item_embedding(item).squeeze(1)

        # Elementwise multiplication
        GMF_layer = embedding1 * embedding2

        # sum GMF layer
        out = torch.sum(GMF_layer, 1).unsqueeze_(1)

        # output between 0 and 1
        out = torch.sigmoid(out)

        return out


class NCF3(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, GMF_model=None, MLP_model=None, device='cuda'):
        super(NCF3, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num).to(device)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num).to(device)

        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1))).to(device)
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1))).to(device)

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout).to(device))
            MLP_modules.append(nn.Linear(input_size, input_size // 2).to(device))
            MLP_modules.append(nn.ReLU().to(device))
        self.MLP_layers = nn.Sequential(*MLP_modules).to(device)

        predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1).to(device)

        self.out_act = nn.Sigmoid().to(device)
        # self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01).to(device)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01).to(device)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01).to(device)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01).to(device)
        #
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight).to(device)

        nn.init.kaiming_uniform_(self.predict_layer.weight,
                                 a=1,
                                 nonlinearity='sigmoid'
                                 ).to(device)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)
        prediction = self.predict_layer(concat)
        # return prediction
        return self.out_act(prediction)


# class NCF(nn.Module):
#     def __init__(self, num_users, num_items, embed_size, num_hidden, output_size):
#         super(NCF, self).__init__()
#         torch.manual_seed(0)
#
#         # user and item embedding layers
#         self.user_emb = nn.Embedding(num_users, embed_size).to(device)
#         self.like_emb = nn.Embedding(num_items, embed_size).to(device)
#
#         self.fc1 = nn.Linear(embed_size * 2, num_hidden[0]).to(device)
#         self.relu1 = nn.ReLU().to(device)
#         self.fc2 = nn.Linear(num_hidden[0], num_hidden[1]).to(device)
#         self.relu2 = nn.ReLU().to(device)
#         self.fc3 = nn.Linear(num_hidden[1], num_hidden[2]).to(device)
#         self.relu3 = nn.ReLU().to(device)
#         self.fc4 = nn.Linear(num_hidden[2], num_hidden[3]).to(device)
#         self.relu4 = nn.ReLU().to(device)
#         self.outLayer = nn.Linear(num_hidden[3], output_size).to(device)
#         self.out_act = nn.Sigmoid().to(device)
#
#     def forward(self, u, v):
#         U = self.user_emb(u)
#         V = self.like_emb(v)
#         print(U.shape)
#         out = torch.cat([U, V], dim=1)
#         out = self.fc1(out)
#         print(out.shape)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         print(out.shape)
#
#         out = self.relu2(out)
#         out = self.fc3(out)
#         print(out.shape)
#
#         out = self.relu3(out)
#         out = self.fc4(out)
#         print(out.shape)
#
#         out = self.relu4(out)
#         print(out.shape)
#
#         out = self.outLayer(out)
#         print(out.shape)
#
#         out = self.out_act(out)
#         print(out.shape)
#
#         return out

