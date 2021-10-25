from tensorflow import keras, nn
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, multiply, concatenate
from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adagrad, Adam
import numpy as np


class Model:
    def __init__(self, n_users, n_items, layers=[20, 10], reg=[0, 0], latent_dim=8):
        self.layers = layers
        self.reg = reg
        self.user_input = Input(shape=(1,), dtype='int32', name='user')
        self.item_input = Input(shape=(1,), dtype='int32', name='movie')

        self.gmf_user_embedding = Embedding(input_dim=n_users,
                                            output_dim=latent_dim,
                                            name='gmf_user_embedding',
                                            embeddings_regularizer=l2(0),
                                            input_length=1)
        self.gmf_item_embedding = Embedding(input_dim=n_items,
                                            output_dim=latent_dim,
                                            name='gmf_item_embedding',
                                            # init = init_normal,
                                            embeddings_regularizer=l2(0),
                                            input_length=1)
        self.mlp_user_embedding = Embedding(input_dim=n_users,
                                            output_dim=latent_dim,
                                            name='mlp_user_embedding',
                                            embeddings_regularizer=l2(0),
                                            input_length=1)
        self.mlp_item_embedding = Embedding(input_dim=n_items,
                                            output_dim=latent_dim,
                                            name='mlp_item_embedding',
                                            embeddings_regularizer=l2(0),
                                            input_length=1)

        self.model = None
        self.num_users = n_users

    def GMF(self):
        user_latent = Flatten()(self.gmf_user_embedding(self.user_input))
        item_latent = Flatten()(self.gmf_item_embedding(self.item_input))
        predict = multiply([user_latent, item_latent])
        prediction = Dense(1, activation=nn.sigmoid, kernel_initializer='lecun_uniform')(predict)
        self.model = keras.Model(
            inputs=[self.user_input, self.item_input],
            outputs=prediction
        )

    def MLP(self):
        user_latent = Flatten()(self.mlp_user_embedding(self.user_input))
        item_latent = Flatten()(self.mlp_item_embedding(self.item_input))
        vector = concatenate([user_latent, item_latent])
        for i in range(len(self.layers)):
            layer = Dense(self.layers[i], kernel_regularizer=l2(self.reg[i]), activation=nn.relu)
            vector = layer(vector)
        prediction = Dense(1, activation=nn.sigmoid, kernel_initializer='lecun_uniform')(vector)
        self.model = keras.Model(
            inputs=[self.user_input, self.item_input],
            outputs=prediction
        )

    def NCF(self):
        gmf_user_latent = Flatten()(self.gmf_user_embedding(self.user_input))
        gmf_item_latent = Flatten()(self.gmf_item_embedding(self.item_input))
        gmf_vector = multiply([gmf_user_latent, gmf_item_latent])

        mlp_user_latent = Flatten()(self.mlp_user_embedding(self.user_input))
        mlp_item_latent = Flatten()(self.mlp_item_embedding(self.item_input))
        mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])

        for i in range(len(self.layers)):
            layer = Dense(self.layers[i], kernel_regularizer=l2(self.reg[i]), activation=nn.relu)
            mlp_vector = layer(mlp_vector)
        predict = concatenate([gmf_vector, mlp_vector])
        prediction = Dense(1, activation=nn.sigmoid, kernel_initializer='lecun_uniform')(predict)
        self.model = keras.Model(
            inputs=[self.user_input, self.item_input],
            outputs=prediction
        )

    def compile_model(self, learning_rate=.001, optimizer='adam'):
        if optimizer == 'adam':
            self.model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        else:
            self.model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        # return self.model

    def train_model(self, X_train, y_train, test, X_test, batch_size, epochs):
        for i in range(epochs):
            self.model.fit(X_train, y_train, batch_size=batch_size, epochs=3, verbose=1, shuffle=True)
            hr, ndcg = self.evaluate(test, X_test, k=10)
            print(f'hit rate:{hr}\nndcg:{ndcg}\n')
        # return self.model

    def evaluate(self, df, X_test, k=10):
        df['score'] = self.model.predict(X_test)
        grouped = df.copy(deep=True)
        grouped['rank'] = grouped.groupby('uid')['score'].rank(method='first', ascending=False)
        grouped.sort_values(['uid', 'rank'], inplace=True)
        top_k = grouped[grouped['rank'] <= k]
        test_in_top_k = top_k[top_k['rating'] == 1]
        hr = test_in_top_k.shape[0] / self.num_users
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: np.log(2) / np.log(1 + x))
        ndcg = test_in_top_k.ndcg.sum() / self.num_users
        return hr, ndcg
