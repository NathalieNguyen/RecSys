import numpy as np
import pandas as pd

from lib.preprocessing import data_for_training as data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == '__main__':
    np.random.seed(0)

    ratings_new = data.match_uid_and_isbn()
    ratings_explicit = ratings_new[ratings_new['Book-Rating'] > 0]

    user_with_threshold = ratings_explicit.groupby('User-ID').count()
    user_with_threshold = user_with_threshold[user_with_threshold['Book-Rating'] >= 2500].index
    isbn_with_threshold = ratings_explicit.groupby('ISBN').count()
    isbn_with_threshold = isbn_with_threshold[isbn_with_threshold['Book-Rating'] >= 300].index
    ratings_above_count_threshold = ratings_explicit[ratings_explicit['User-ID'].isin(user_with_threshold)]
    ratings_above_count_threshold = ratings_explicit[ratings_explicit['ISBN'].isin(isbn_with_threshold)]

    train_set, test_set = train_test_split(ratings_above_count_threshold, test_size=0.1)
    ratings_copy = train_set.copy()

    user_series = pd.Series(ratings_above_count_threshold['User-ID'].unique())
    for user_index, user_id in user_series.iteritems():
        print("Processing user:", user_id)
        isbn_df = pd.DataFrame(ratings_above_count_threshold['ISBN']).drop_duplicates()
        ratings_by_user = pd.DataFrame(ratings_above_count_threshold[ratings_above_count_threshold['User-ID'] == user_id])
        isbn_from_test_set = pd.DataFrame(test_set[test_set['User-ID'] == user_id]['ISBN'])
        isbn_not_rated_by_user = isbn_df[~isbn_df['ISBN'].isin(ratings_by_user['ISBN'])]
        isbn_not_rated_by_user = isbn_not_rated_by_user.append(isbn_from_test_set)

        for isbn_index, isbn_row in isbn_not_rated_by_user.iterrows():
            isbn_for_prediction = isbn_row['ISBN']

            users_for_knn = ratings_above_count_threshold[ratings_above_count_threshold['ISBN'] == isbn_for_prediction][
                'User-ID']
            ratings_by_other_users = ratings_above_count_threshold[ratings_above_count_threshold['User-ID'].isin(users_for_knn)]
            ratings_for_knn = pd.concat([ratings_by_user, ratings_by_other_users]).drop_duplicates()
            ratings_pivot = ratings_for_knn.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

            item_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=min(users_for_knn.count(), 5) + 1)
            item_knn.fit(ratings_pivot)
            distances, indices = item_knn.kneighbors(ratings_pivot.loc[user_id, :].values.reshape(1, -1))

            knn_ratings = ratings_pivot.iloc[indices.flatten()[1:], :]

            predicted_mean_rating = round(knn_ratings[isbn_for_prediction].mean())

            similarities = 1 - distances
            predicted_weighted_avg = round(np.average(knn_ratings[isbn_for_prediction], weights=similarities.flatten()[1:]))

            predicted_ratings = pd.DataFrame([[user_id, isbn_for_prediction, round(predicted_weighted_avg)]],
                                             columns=list(ratings_copy.columns))
            ratings_copy = ratings_copy.append(predicted_ratings, ignore_index=True)
    print(ratings_copy)

    actual_ratings = []
    predicted_ratings = []
    for index, row in test_set.iterrows():
        actual_ratings.append(row['Book-Rating'])
        predicted_rating = ratings_copy[(ratings_copy['User-ID'] == row['User-ID']) & (ratings_copy['ISBN'] == row['ISBN'])]
        predicted_ratings.append(predicted_rating.iloc[0,2])

    print('actual_ratings', actual_ratings)
    print('predicted_ratings', predicted_ratings)
    RMSE_test_set = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print('RMSE test set: ', RMSE_test_set)
