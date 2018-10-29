from lib.preprocessing import data_for_training as data
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

    # load ratings above count threshold
    ratings_above_count_threshold_df = data.densify_ratings_df(user_ratings_count_threshold=50,
                                                               isbn_ratings_count_threshold=200)

    # create test set and copy of original ratings
    train_set, test_set = train_test_split(ratings_above_count_threshold_df, test_size=0.1)
    ratings_copy = train_set.copy()

    mean_rating = ratings_above_count_threshold_df['Book-Rating'].mean()

    test_set_without_ratings = test_set.copy()
    test_set_without_ratings['Book-Rating'] = np.nan

    full_set_without_test_set_ratings = pd.concat([train_set, test_set_without_ratings])

    ratings_pivot = full_set_without_test_set_ratings.pivot(index='User-ID', columns='ISBN',
                                                            values='Book-Rating').fillna(mean_rating)

    # decomposition and prediction
    R_pivot = ratings_pivot.values
    sparse_R = csr_matrix(R_pivot, dtype=float)

    U, sigma, Vt = svds(sparse_R, k=20)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(U, np.dot(sigma, Vt))

    all_user_predicted_ratings_df = pd.DataFrame(all_user_predicted_ratings)
    all_user_predicted_ratings_df.set_axis(ratings_pivot.columns, axis=1, inplace=True)
    all_user_predicted_ratings_df.set_axis(ratings_pivot.index, axis=0, inplace=True)

    # making recommendations
    recs_predicted_ratings_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_pivot.columns).set_index(
        ratings_pivot.index).sample(n=1)

    isbn_book_dict = recs.map_isbn_to_names()

    for user_id, row in recs_predicted_ratings_df.iterrows():
        topn_results = row.sort_values(ascending=False).iloc[:5]
        print('Finding recommendations for user {0}...'.format(user_id))

        rec_index = 1
        for rec_isbn, rec_score in topn_results.iteritems():
            rec_book_name = isbn_book_dict[rec_isbn]
            print('Top {0} ISBN: {1}, {2}'.format(rec_index, rec_isbn, rec_book_name))
            rec_index += 1

    # metrics
    actual_ratings = []
    predicted_ratings = []
    for index, row in test_set.iterrows():
        actual_ratings.append(row['Book-Rating'])
        predicted_ratings.append(all_user_predicted_ratings_df.at[row['User-ID'], row['ISBN']])

    RMSE_train_set = sqrt(mean_squared_error(R_pivot, all_user_predicted_ratings))
    print('RMSE train set: ', RMSE_train_set)

    RMSE_test_set = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print('RMSE test set: ', RMSE_test_set)
