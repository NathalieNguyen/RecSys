from lib.preprocessing import data_for_training as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from math import sqrt


if __name__ == '__main__':
    np.random.seed(0)

    ratings_above_count_threshold_df = data.filter_ratings_above_count_threshold(frac=0.1)
    train_set, test_set = train_test_split(ratings_above_count_threshold_df, test_size=0.1,
                                           stratify=ratings_above_count_threshold_df['User-ID'])
    mean_rating = ratings_above_count_threshold_df['Book-Rating'].mean()

    test_set_without_ratings = test_set.copy()
    test_set_without_ratings['Book-Rating'] = np.nan

    full_set_without_test_set_ratings = pd.concat([train_set, test_set_without_ratings])

    ratings_pivot = full_set_without_test_set_ratings.pivot(index='User-ID', columns='ISBN',
                                                            values='Book-Rating').fillna(mean_rating)

    R_pivot = ratings_pivot.values

    model = NMF(n_components=20, init='random', random_state=0)
    W = model.fit_transform(R_pivot)
    H = model.components_
    all_user_predicted_ratings = np.dot(W, H)

    RMSE_full_set = sqrt(mean_squared_error(R_pivot, all_user_predicted_ratings))
    print('RMSE full set: ', RMSE_full_set)

    all_user_predicted_ratings_df = pd.DataFrame(all_user_predicted_ratings)
    all_user_predicted_ratings_df.set_axis(ratings_pivot.columns, axis=1, inplace=True)
    all_user_predicted_ratings_df.set_axis(ratings_pivot.index, axis=0, inplace=True)

    actual_ratings = []
    predicted_ratings = []
    for index, row in test_set.iterrows():
        actual_ratings.append(row['Book-Rating'])
        predicted_ratings.append(all_user_predicted_ratings_df.at[row['User-ID'], row['ISBN']])

    RMSE_test_set = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print('RMSE test set: ', RMSE_test_set)
