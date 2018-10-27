from lib.preprocessing import data_for_training as data
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np


if __name__ == '__main__':

    ratings_explicit, _ = data.separate_explicit_and_implicit_ratings()
    ratings_explicit = ratings_explicit

    ratings_explicit_copy = ratings_explicit.copy()
    ratings_explicit_copy['Book-Rating'] = np.nan

    ratings_df_random = pd.DataFrame(np.random.randint(1, 11, size=ratings_explicit.shape),
                                     index=ratings_explicit.index, columns=ratings_explicit.columns)

    ratings_filled_with_random = ratings_explicit_copy.fillna(ratings_df_random)

    actual_ratings = ratings_explicit['Book-Rating']
    predicted_ratings = ratings_filled_with_random['Book-Rating']

    RMSE = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print('RMSE: ', RMSE)
