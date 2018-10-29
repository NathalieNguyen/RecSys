from lib.preprocessing import data_for_training as data
from lib.evaluation import recommendations as recs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == '__main__':
    np.random.seed(0)

    # load ratings above count threshold
    ratings_above_count_threshold = data.densify_ratings_df(user_ratings_count_threshold=50,
                                                            isbn_ratings_count_threshold=200)

    # create test set and copy of original ratings
    train_set, test_set = train_test_split(ratings_above_count_threshold, test_size=0.1)
    ratings_copy = train_set.copy()

    # compute cosine similarity between each user and every other user who rated the book we are looking at
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

            item_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=min(users_for_knn.count(), 20))
            item_knn.fit(ratings_pivot)
            distances, indices = item_knn.kneighbors(ratings_pivot.loc[user_id, :].values.reshape(1, -1))

            knn_ratings = ratings_pivot.iloc[indices.flatten()[1:], :]

            # making predictions
            similarities = 1 - distances
            if similarities.flatten()[1:].sum() > 0:
                predicted_rating = np.average(knn_ratings[isbn_for_prediction], weights=similarities.flatten()[1:])
            else:
                predicted_rating = knn_ratings[isbn_for_prediction].mean()

            predicted_ratings = pd.DataFrame([[user_id, isbn_for_prediction, round(predicted_rating)]],
                                             columns=list(ratings_copy.columns))
            ratings_copy = ratings_copy.append(predicted_ratings, ignore_index=True)

    # making recommendations
    recs_predicted_ratings_df = ratings_copy.pivot(index='User-ID', columns='ISBN', values='Book-Rating').sample(n=1)

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
        predicted_rating = ratings_copy[(ratings_copy['User-ID'] == row['User-ID']) & (ratings_copy['ISBN'] == row['ISBN'])]
        predicted_ratings.append(predicted_rating.iloc[0, 2])

    RMSE_test_set = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print('RMSE test set: ', RMSE_test_set)
