import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from lib.models import neural_network_structure as nns
from lib.preprocessing import data_for_training as data


if __name__ == '__main__':
    np.random.seed(0)

    # load ratings and take sample
    ratings_new = data.match_uid_and_isbn()
    ratings_sample = ratings_new.sample(n=10000)
    n_ratings = len(ratings_sample)

    # create user lookup
    users_unique = ratings_sample['User-ID'].unique()
    n_users = len(users_unique)

    user_df = pd.DataFrame(users_unique, columns=['user_id'])
    user_df['idx'] = user_df.index
    user_idx_lookup = user_df.set_index('user_id')

    # create item lookup
    items_unique = ratings_sample['ISBN'].unique()
    n_items = len(items_unique)

    items_df = pd.DataFrame(items_unique, columns=['isbn'])
    items_df['idx'] = items_df.index
    items_idx_lookup = items_df.set_index('isbn')

    # transform sample data set with idx
    ratings_sample_transformed = ratings_sample.copy()
    ratings_sample_transformed.loc[:, "User-ID"] = ratings_sample["User-ID"].apply(lambda x: user_idx_lookup.loc[x, "idx"])
    ratings_sample_transformed.loc[:, "ISBN"] = ratings_sample["ISBN"].apply(lambda x: items_idx_lookup.loc[x, "idx"])

    rating_tensor = torch.Tensor(ratings_sample_transformed.values).long()

    # neural network parameters
    dtype = torch.long
    device = torch.device('cpu')
    batch_size = 100
    learning_rate = 0.005

    # instantiate neural network
    model = nns.RecommenderNet(n_users, n_items, 1024, 128, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_hist = []

    # Train
    x_user_onehot = torch.FloatTensor(batch_size, n_users)
    x_item_onehot = torch.FloatTensor(batch_size, n_items)

    n_batches = int(n_ratings / batch_size)

    for epoche in range(100):
        for b in range(n_batches):
            rating_batch = rating_tensor[b * batch_size: (b + 1) * batch_size]

            x_user_onehot.zero_()
            x_user_onehot.scatter_(1, rating_batch[:, 0].reshape(-1, 1), 1)

            x_item_onehot.zero_()
            x_item_onehot.scatter_(1, rating_batch[:, 1].reshape(-1, 1), 1)

            y = rating_batch[:, 2].float().reshape(-1, 1)

            # forward step
            outputs = model.forward(x_user_onehot, x_item_onehot)

            # compute loss
            loss = criterion(outputs, y)

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute error
        if epoche % 10 == 0:
            loss_hist.append(loss.item())
            print(epoche, loss.item())
