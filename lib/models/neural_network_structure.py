import torch
import torch.nn as nn
import torch.nn.functional as F


class RecommenderNet(nn.Module):

    def __init__(self, n_users, n_items, H1, H2, D_out):
        super(RecommenderNet, self).__init__()

        # linear layers
        self.linear1 = nn.Linear(n_users + n_items, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    def forward(self, users, items):
        x = torch.cat((users, items), 1)
        h1_relu = F.relu(self.linear1(x))
        h2_relu = F.relu(self.linear2(h1_relu))
        output_scores = self.linear3(h2_relu)
        return output_scores
