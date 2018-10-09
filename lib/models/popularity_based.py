from lib.preprocessing import books as books
from lib.preprocessing import data_for_training as data
import pandas as pd


def build_popularity_based_recommendations():
    ratings_explicit, ratings_implicit = data.separate_explicit_and_implicit_ratings()
    ratings_count = pd.DataFrame(ratings_explicit.groupby(['ISBN'])['Book-Rating'].sum())
    top10 = ratings_count.sort_values('Book-Rating', ascending=False).head(10)
    return top10.merge(books.load_cleaned_df(), left_on='ISBN', right_on='ISBN', how='left')
