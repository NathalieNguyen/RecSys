from lib.preprocessing import data_for_training as data
from lib.preprocessing import books as books
from lib.evaluation import recommendations as recs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt


if __name__ == '__main__':
    np.random.seed(0)

    stratify_df = data.filter_ratings_above_count_threshold(frac=0.1)
    train_set, test_set = train_test_split(stratify_df, stratify=stratify_df['User-ID'])
    print(train_set)

    test_set_without_ratings = test_set.copy()
    test_set_without_ratings['Book-Rating'] = np.nan

    full_set_without_test_set_ratings = pd.concat([train_set, test_set_without_ratings])

    ratings_pivot = full_set_without_test_set_ratings.pivot(index='User-ID', columns='ISBN',
                                                            values='Book-Rating').fillna(0)
    print(ratings_pivot.max())

    R_pivot = ratings_pivot.values
    # user_ratings_mean = np.mean(R_pivot, axis=1)
    # R_demeaned = R_pivot - user_ratings_mean.reshape(-1, 1)
    sparse_R = csr_matrix(R_pivot, dtype=float)

    U, sigma, Vt = svds(sparse_R, k=5)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(U, np.dot(sigma, Vt))

    print("U: ", U)
    print("sigma: ", sigma)
    print("Vt: ", Vt)

    RMSE_full_set = sqrt(mean_squared_error(ratings_pivot.values, all_user_predicted_ratings))
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
