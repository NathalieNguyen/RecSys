import pandas as pd
import numpy as np


def load_raw_df():
    return pd.read_csv('../data/BX-CSV-Dump/BX-Books.csv',
                       sep=';', error_bad_lines=False, encoding='latin-1', low_memory=False)


def load_cleaned_df():
    books_raw = load_raw_df()
    books_cleaned = fix_wrong_columns_for_entries(books_raw)
    books_cleaned = adjust_year_of_publication(books_cleaned)
    return fill_nan(books_cleaned)


def fix_wrong_columns_for_entries(books):
    books.loc[books['ISBN'] == '078946697X', 'Year-Of-Publication'] = 2000
    books.loc[books['ISBN'] == '078946697X', 'Book-Author'] = 'Michael Teitelbaum'
    books.loc[books['ISBN'] == '078946697X', 'Publisher'] = 'DK Publishing Inc'
    books.loc[books['ISBN'] == '078946697X', 'Book-Title'] = 'DK Readers: Creating the X-Men, How It All Began ' \
                                                             '(Level 4: Proficient Readers)'

    books.loc[books['ISBN'] == '0789466953', 'Year-Of-Publication'] = 2000
    books.loc[books['ISBN'] == '0789466953', 'Book-Author'] = 'James Buckley'
    books.loc[books['ISBN'] == '0789466953', 'Publisher'] = 'DK Publishing Inc'
    books.loc[books['ISBN'] == '0789466953', 'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books ' \
                                                             'Come to Life (Level 4: Proficient Readers)'

    books.loc[books['ISBN'] == '2070426769', 'Year-Of-Publication'] = 2003
    books.loc[books['ISBN'] == '2070426769', 'Book-Author'] = 'Jean-Marie Gustave Le Clezio'
    books.loc[books['ISBN'] == '2070426769', 'Publisher'] = 'Gallimard'
    books.loc[books['ISBN'] == '2070426769', 'Book-Title'] = 'Peuple Du Ciel Suivi de les Bergers'
    return books


def adjust_year_of_publication(books):
    books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
    books.loc[(books['Year-Of-Publication'] > 2018) |
              (books['Year-Of-Publication'] == 0), 'Year-Of-Publication'] = np.nan
    books['Year-Of-Publication'].fillna(round(books['Year-Of-Publication'].mean()), inplace=True)
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(np.int32)
    return books


def fill_nan(books):
    books.loc[books['ISBN'] == '193169656X', 'Publisher'] = 'NovelBooks'
    books.loc[books['ISBN'] == '1931696993', 'Publisher'] = 'CreateSpace Independent Publishing Platform'
    books.loc[books['ISBN'] == '9627982032', 'Book-Author'] = 'other'
    return books
