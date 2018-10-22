from lib.preprocessing import books as books
from lib.preprocessing import users as users
from lib.preprocessing import ratings as ratings
from lib.features import categories as categories
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def compute_genres_matrix():
    """Compute genres matrix out of binarized book categories"""
    genres_matrix_columns = categories.binarize_books_categories().columns
    return categories.merge_books_with_categories()[genres_matrix_columns]


def match_uid_and_isbn():
    """Check if ratings data have User-ID and ISBN which exist in respective tables users and books"""
    books_df = books.load_cleaned_df()
    users_df = users.load_cleaned_df()
    ratings_df = ratings.load_raw_df()
    ratings_new = ratings_df[ratings_df['ISBN'].isin(books_df['ISBN'])]
    return ratings_new[ratings_new['User-ID'].isin(users_df['User-ID'])]


def separate_explicit_and_implicit_ratings():
    """Separate explicit and implicit ratings for data exploration"""
    ratings_new = match_uid_and_isbn()
    ratings_explicit = ratings_new[ratings_new['Book-Rating'] > 0]
    ratings_implicit = ratings_new[ratings_new['Book-Rating'] == 0]
    return ratings_explicit, ratings_implicit


def filter_ratings_above_count_threshold(frac=1):
    """Filter users with explicit ratings count above threshold and take a fraction for training the models"""
    ratings_new = match_uid_and_isbn()
    ratings_explicit = ratings_new[ratings_new['Book-Rating'] > 0]
    grouped = ratings_explicit.groupby('User-ID').count().reset_index().sample(frac=frac)
    filtered = grouped[grouped['Book-Rating'] >= 2].rename(index=str, columns={'ISBN': 'ISBN count',
                                                                               'Book-Rating': 'Book-Rating count'})
    ratings_above_count_threshold = ratings_explicit[ratings_explicit['User-ID'].isin(filtered['User-ID'])]
    return ratings_above_count_threshold


def filter_ratings_below_count_threshold():
    """Filter users with explicit ratings count below threshold for training content-based model"""
    ratings_new = match_uid_and_isbn()
    ratings_above_threshold = filter_ratings_above_count_threshold(frac=1)
    return ratings_new[~ratings_new['User-ID'].isin(ratings_above_threshold['User-ID'])]


def build_train_test():
    """Build train test data based on ratings above count threshold"""
    ratings_above_count_threshold_df = filter_ratings_above_count_threshold().sample(frac=0.1)
    train_set, test_set = train_test_split(ratings_above_count_threshold_df, test_size=0.1,
                                           stratify=ratings_above_count_threshold_df['User-ID'])
    return train_set, test_set


def build_full_set_with_hidden_ratings():
    """Build full set with hidden ratings from the defined test set"""
    train_set, test_set = build_train_test()
    test_set_without_ratings = test_set.copy()
    test_set_without_ratings['Book-Rating'] = np.nan
    return pd.concat([train_set, test_set_without_ratings])


def build_ratings_pivot_with_fillna_mean():
    """Build ratings pivot and fill nan with mean"""
    pass

