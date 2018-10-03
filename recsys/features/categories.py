import pandas as pd
import recsys.preprocessing.books


def load_books_categories():
    return pd.read_json('data/books_categories.json')


def binarize_books_categories():
    books_categories = load_books_categories()
    books_categories_dropped_duplicates = books_categories.drop_duplicates()
    books_categories_binarized = pd.get_dummies(books_categories_dropped_duplicates,
                                                columns=['category'], prefix='').groupby(['ISBN'], as_index=False).sum()
    return books_categories_binarized.drop(books_categories_binarized.columns[1], axis=1)


def merge_books_with_categories():
    books = recsys.preprocessing.books.load_cleaned_df()
    return books.merge(binarize_books_categories(), left_on='ISBN', right_on='ISBN', how='inner')
