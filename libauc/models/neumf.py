# This implementation is from https://github.com/hexiangnan/neural_collaborative_filtering

import torch
import torch.nn as nn
import logging

class NeuMF(nn.Module):
    r"""
        NeuMF is a widely-used model for recommender systems.

        args:
            user_num (int): the number of users in the dataset
            item_num (int): the number of items in the dataset
            dropout (float, optional): dropout ratio for the model
            emb_size (int, optional): embedding size of the model
            layers (string, optional): describe the layer information of the model

        Reference:
            .. [1] He, X., Liao, L., Zhang, H., Nie, L., Hu, X., and Chua, T.
                    Neural Collaborative Filtering
                    https://arxiv.org/abs/1708.05031
    """
    def __init__(self, user_num: int, item_num: int, dropout: float=0.2, emb_size: int=64, layers: str='[64]'):
        super(NeuMF, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.emb_size = emb_size
        self.dropout = dropout
        self.layers = eval(layers)

        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.mlp_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mlp_i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.emb_size
        for i, layer_size in enumerate(self.layers):
            self.mlp.append(nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.prediction = nn.Linear(pre_size + self.emb_size, 1, bias=False)

    def reset_last_layer(self):
        self.prediction.reset_parameters()

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        logging.info('Load model from ' + model_path)

    def forward(self, feed_dict):
        u_ids = feed_dict['user_id'].long()  # [batch_size]
        i_ids = feed_dict['item_id'].long()  # [batch_size, -1]

        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        mlp_u_vectors = self.mlp_u_embeddings(u_ids)
        mlp_i_vectors = self.mlp_i_embeddings(i_ids)

        mf_vector = mf_u_vectors * mf_i_vectors
        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}