# %%
# Collaborative Filtering
# use embedding to build a simple recommendation system
# Source:
# 1. Collaborative filtering, https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb
# 2. https://github.com/yanneta/pytorch-tutorials/blob/master/collaborative-filtering-nn.ipynb
# training neural network based collaborative filtering
# neural network model (NCF)

import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_size, num_hidden, output_size):
        super(NCF, self).__init__()
        torch.manual_seed(0)

        # user and item embedding layers
        self.user_embedding = nn.Embedding(num_users, embed_size).to(device)
        self.item_embedding = nn.Embedding(num_items, embed_size).to(device)

        self.fc1 = nn.Linear(embed_size * 2, num_hidden[0]).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1]).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.fc3 = nn.Linear(num_hidden[1], num_hidden[2]).to(device)
        self.relu3 = nn.ReLU().to(device)
        self.fc4 = nn.Linear(num_hidden[2], num_hidden[3]).to(device)
        self.relu4 = nn.ReLU().to(device)
        self.outLayer = nn.Linear(num_hidden[3], output_size).to(device)
        self.out_act = nn.Sigmoid().to(device)

    def forward(self, u, v):
        U = self.user_embedding(u)
        V = self.item_embedding(v)
        out = torch.cat([U, V], dim=1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.outLayer(out)
        out = self.out_act(out)
        return out



# from tensorflow import keras, nn
# from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, multiply, concatenate
# from tensorflow.keras.regularizers import l2
#
# from tensorflow.keras.optimizers import Adagrad, Adam
# import numpy as np


# class Model:
#     def __init__(self, n_users, n_items, layers=[20, 10], reg=[0, 0], latent_dim=8):
#         self.layers = layers
#         self.reg = reg
#         self.user_input = Input(shape=(1,), dtype='int32', name='user')
#         self.item_input = Input(shape=(1,), dtype='int32', name='movie')
#
#         self.gmf_user_embedding = Embedding(input_dim=n_users,
#                                             output_dim=latent_dim,
#                                             name='gmf_user_embedding',
#                                             embeddings_initializer='uniform',
#                                             embeddings_regularizer=l2(0),
#                                             input_length=1)
#         self.gmf_item_embedding = Embedding(input_dim=n_items,
#                                             output_dim=latent_dim,
#                                             name='gmf_item_embedding',
#                                             embeddings_initializer='uniform',
#                                             embeddings_regularizer=l2(0),
#                                             input_length=1)
#         self.mlp_user_embedding = Embedding(input_dim=n_users,
#                                             output_dim=latent_dim,
#                                             name='mlp_user_embedding',
#                                             embeddings_initializer='uniform',
#                                             embeddings_regularizer=l2(0),
#                                             input_length=1)
#         self.mlp_item_embedding = Embedding(input_dim=n_items,
#                                             output_dim=latent_dim,
#                                             name='mlp_item_embedding',
#                                             embeddings_initializer='uniform',
#                                             embeddings_regularizer=l2(0),
#                                             input_length=1)
#
#         self.model = None
#         self.num_users = n_users
#
#     def GMF(self):
#         user_latent = Flatten()(self.gmf_user_embedding(self.user_input))
#         item_latent = Flatten()(self.gmf_item_embedding(self.item_input))
#         predict = multiply([user_latent, item_latent])
#         prediction = Dense(1, activation=nn.sigmoid, kernel_initializer='lecun_uniform')(predict)
#         self.model = keras.Model(
#             inputs=[self.user_input, self.item_input],
#             outputs=prediction
#         )
#
#     def MLP(self):
#         user_latent = Flatten()(self.mlp_user_embedding(self.user_input))
#         item_latent = Flatten()(self.mlp_item_embedding(self.item_input))
#         vector = concatenate([user_latent, item_latent])
#         for i in range(len(self.layers)):
#             layer = Dense(self.layers[i], kernel_regularizer=l2(self.reg[i]), activation=nn.relu)
#             vector = layer(vector)
#         prediction = Dense(1, activation=nn.sigmoid, kernel_initializer='lecun_uniform')(vector)
#         self.model = keras.Model(
#             inputs=[self.user_input, self.item_input],
#             outputs=prediction
#         )
#
#     def NCF(self):
#         gmf_user_latent = Flatten()(self.gmf_user_embedding(self.user_input))
#         gmf_item_latent = Flatten()(self.gmf_item_embedding(self.item_input))
#         gmf_vector = multiply([gmf_user_latent, gmf_item_latent])
#
#         mlp_user_latent = Flatten()(self.mlp_user_embedding(self.user_input))
#         mlp_item_latent = Flatten()(self.mlp_item_embedding(self.item_input))
#         mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
#
#         for i in range(len(self.layers)):
#             layer = Dense(self.layers[i], kernel_regularizer=l2(self.reg[i]), activation=nn.relu)
#             mlp_vector = layer(mlp_vector)
#         predict = concatenate([gmf_vector, mlp_vector])
#         prediction = Dense(1, activation=nn.sigmoid, kernel_initializer='lecun_uniform')(predict)
#         self.model = keras.Model(
#             inputs=[self.user_input, self.item_input],
#             outputs=prediction
#         )
#
#     def compile_model(self, learning_rate=.001, optimizer='adam'):
#         if optimizer == 'adam':
#             self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
#         else:
#             self.model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
#         # return self.model
#
#     def train_model(self, X_train, y_train, test, X_test, batch_size, epochs):
#         for i in range(epochs):
#             hist = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
#             hr, ndcg = self.evaluate(test, X_test, k=10)
#             print(f'hit rate:{hr}\nndcg:{ndcg}\n')
#         # return self.model
#
#     def evaluate(self, df, X_test, k=10):
#         df['score'] = self.model.predict(X_test)
#         grouped = df.copy(deep=True)
#         grouped['rank'] = grouped.groupby('uid')['score'].rank(method='first', ascending=False)
#         grouped.sort_values(['uid', 'rank'], inplace=True)
#         top_k = grouped[grouped['rank'] <= k]
#         test_in_top_k = top_k[top_k['rating'] == 1]
#         hr = test_in_top_k.shape[0] / self.num_users
#         test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: np.log(2) / np.log(1 + x))
#         ndcg = test_in_top_k.ndcg.sum() / self.num_users
#         return hr, ndcg
