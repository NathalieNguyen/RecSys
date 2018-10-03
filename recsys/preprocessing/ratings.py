import pandas as pd


def load_raw_df():
    return pd.read_csv('../data/BX-CSV-Dump/BX-Book-Ratings.csv',
                       sep=';', error_bad_lines=False, encoding='latin-1', low_memory=False)
