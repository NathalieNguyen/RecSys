import pandas as pd
import numpy as np


def load_raw_df():
    return pd.read_csv('lib/data/BX-CSV-Dump/BX-Users.csv',
                       sep=';', error_bad_lines=False, encoding='latin-1', low_memory=False)


def load_cleaned_df():
    users_raw = load_raw_df()
    return adjust_age(users_raw)


def adjust_age(users):
    users.loc[(users['Age'] < 5) | (users['Age'] > 90), 'Age'] = np.nan
    users['Age'].fillna(users['Age'].mean(), inplace=True)
    users['Age'] = users['Age'].astype(np.int32)
    return users
