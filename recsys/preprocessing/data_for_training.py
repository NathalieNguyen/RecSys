from recsys.preprocessing import books as books
from recsys.preprocessing import users as users
from recsys.preprocessing import ratings as ratings
from recsys.features import categories as categories
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split


def compute_genres_matrix():
    """Compute genres matrix out of binarized book categories"""
    genres_matrix_columns = categories.binarize_books_categories().columns
    return categories.merge_books_with_categories()[genres_matrix_columns]


def densify_ratings_df():
    """Densify counts of user and book ratings by condition"""
    ratings_df = ratings.load_raw_df()
    user_ratings_counts = ratings_df['User-ID'].value_counts()
    user_ratings_densified = ratings_df[ratings_df['User-ID'].isin(user_ratings_counts[user_ratings_counts >= 50].index)]
    book_ratings_counts = ratings_df['ISBN'].value_counts()
    return user_ratings_densified[user_ratings_densified['ISBN'].isin(book_ratings_counts[book_ratings_counts >= 100].index)]


def split_ratings_train_test(df):
    """After splitting into train and test data choose extract method"""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader)
    train_set, test_set = train_test_split(data, test_size=0.25, random_state=1)
    return train_set, test_set
