import torch
import torch.nn as nn


class RecommenderNet(nn.Module):

    def __init__(self, n_users, n_items, n_factors, H1, D_out):
        super(RecommenderNet, self).__init__()
        # user and item embedding layers
        self.user_factors = nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=True)
        # linear layers
        self.linear1 = nn.Linear(n_factors * 2, H1)
        self.linear2 = nn.Linear(H1, D_out)

    def forward(self, users, items):
        users_embedding = self.user_factors(users)
        items_embedding = self.item_factors(items)
        # concatenate user and item embeddings to form input
        x = torch.cat([users_embedding, items_embedding], 1)
        h1_relu = nn.ReLU(self.linear1(x))
        output_scores = self.linear2(h1_relu)
        return output_scores

    def predict(self, users, items):
        # return the score
        output_scores = self.forward(users, items)
        return output_scores
