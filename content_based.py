from lib.preprocessing import data_for_training as data
from lib.evaluation import recommendations as recs
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csc_matrix
import numpy as np
import pandas as pd


if __name__ == '__main__':
    np.random.seed(0)

    # load ratings below count threshold
    below_count = data.filter_ratings_below_count_threshold()
    explicit_ratings_above_6 = below_count[below_count['Book-Rating'] > 6]

    # create genre matrix for computing similarity between items
    genre_matrix = data.compute_genres_matrix()
    genre_matrix_without_ISBN = genre_matrix.drop('ISBN', axis=1)
    genre_sparse_matrix = csc_matrix(genre_matrix_without_ISBN)

    for index, row in explicit_ratings_above_6.sample(n=10).iterrows():
        user_id = row['User-ID']
        isbn = row['ISBN']
        book_name = isbn_book_dict[isbn]
        print('Finding recommendations for user {0}, who has read ISBN: {1}, {2}...'.format(user_id, isbn, book_name))

        # select genre from the current book we are looking at
        book_categories = genre_matrix[genre_matrix['ISBN'] == isbn]

        # remove ISBN column and transform it into a sparse matrix
        book_categories_without_ISBN = book_categories.drop('ISBN', axis=1)
        book_categories_sparse_matrix = csc_matrix(book_categories_without_ISBN)

        # optimize performance by just computing cosine_similarity on X*Y shape (270k, 1) instead of pairwise X*X shape (270k, 270k)
        # where X contains genre information for all 270k entries
        # and Y only contains genre information for the 1 entry we need to look at
        results = cosine_similarity(X=genre_sparse_matrix, Y=book_categories_sparse_matrix)

        # transform ndarray from cosine_similarity into easy-to-access dataframe
        results_df = pd.DataFrame(results)
        results_df.set_axis([isbn], axis=1, inplace=True)
        results_df.set_axis(genre_matrix['ISBN'], inplace=True)
        top_n_results = results_df.sort_values(by=isbn, ascending=False).reset_index().truncate(after=4)

        # making recommendations
        isbn_book_dict = recs.map_isbn_to_names()

        for rec_index, rec_row in top_n_results.iterrows():
            rec_isbn = rec_row['ISBN']
            rec_book_name = isbn_book_dict[rec_isbn]
            print('Top {0} ISBN: {1}, {2}'.format(rec_index + 1, rec_isbn, rec_book_name))
