from lib.preprocessing import books as books


def map_isbn_to_names():
    """Create a dictionary that maps each ISBN to its book title"""
    books_df = books.load_cleaned_df()
    return dict(zip(books_df['ISBN'], books_df['Book-Title']))
