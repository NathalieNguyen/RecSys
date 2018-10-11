from collections import defaultdict
from lib.preprocessing import books as books


def get_topn_recommendations(recs_pred, topN=3):
    """Get n recommendations (only isbn without names)"""
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in recs_pred:
        top_recs[uid].append((iid, est))

    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recs[uid] = user_ratings[:topN]
    return top_recs


def map_isbn_to_names():
    """Create a dictionary that maps each ISBN to its book title"""
    books_df = books.load_cleaned_df()
    return dict(zip(books_df['ISBN'], books_df['Book-Title']))


def show_recommendations(recs_pred, user_id=None):
    """Show recommendations for all users or for one specific user"""
    topn_recommendations = get_topn_recommendations(recs_pred)
    isbn_to_name = map_isbn_to_names()

    if user_id is None:
        for uid, user_ratings in topn_recommendations.items():
            print(uid, [isbn_to_name[iid] for (iid, _) in user_ratings])
    else:
        user_ratings = topn_recommendations[user_id]
        print(user_id, [isbn_to_name[iid] for (iid, _) in user_ratings])