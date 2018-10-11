from lib.preprocessing import books as books
from lib.preprocessing import users as users
from lib.preprocessing import ratings as ratings
from lib.features import categories as categories
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split


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
    ratings_explicit = ratings_new[ratings_new['Book-Rating'] != 0]
    ratings_implicit = ratings_new[ratings_new['Book-Rating'] == 0]
    return ratings_explicit, ratings_implicit


def densify_ratings_df(user_ratings_threshold=50, book_ratings_threshold=100):
    """Densify counts of user and book ratings by condition"""
    ratings_df = match_uid_and_isbn()
    user_ratings_counts = ratings_df['User-ID'].value_counts()
    user_ratings_densified = ratings_df[ratings_df['User-ID'].isin(user_ratings_counts[user_ratings_counts >=
                                                                                       user_ratings_threshold].index)]
    book_ratings_counts = ratings_df['ISBN'].value_counts()
    return user_ratings_densified[user_ratings_densified['ISBN'].isin(book_ratings_counts[book_ratings_counts >=
                                                                                          book_ratings_threshold].index)]


def load_split_ratings_train_test(df):
    """After splitting into train and test set choose extract method (densified ratings df or sample)"""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader)
    train_set, test_set = train_test_split(data, test_size=0.25, random_state=1)
    return train_set, test_set


def load_full_ratings_train_set(df):
    """After loading full train set choose extract method (densified ratings df or sample)"""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader)
    return data.build_full_trainset()
