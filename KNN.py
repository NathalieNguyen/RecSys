from lib.preprocessing import books as books
from lib.preprocessing import data_for_training as data
from surprise import KNNWithMeans
from surprise import accuracy
from collections import defaultdict


if __name__ == '__main__':
    # Choose either densified ratings df or sample for building the train set
    ratings_df_param_sample = data.match_uid_and_isbn().sample(n=10000)
    ratings_df_param_densified = data.densify_ratings_df()

    # Training and testing based on full data set (with anti test set for building recommendations)
    full_train_set = data.load_full_ratings_train_set(ratings_df_param_densified)

    algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
    algo.fit(full_train_set)

    anti_test_set = full_train_set.build_anti_testset()
    recs_pred = algo.test(anti_test_set)
    # print('recs_pred: ', recs_pred)

    def get_top3_recommendations(recs_pred, topN=3):
        """Get n recommendations (only isbn without names)"""
        top_recs = defaultdict(list)
        for uid, iid, true_r, est, _ in recs_pred:
            top_recs[uid].append((iid, est))

        for uid, user_ratings in top_recs.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_recs[uid] = user_ratings[:topN]

        return top_recs


    # print('recs: ', get_top3_recommendations(recs_pred))

    def map_isbn_to_names():
        """Create a dictionary that maps each ISBN to its book title"""
        books_df = books.load_cleaned_df()
        return dict(zip(books_df['ISBN'], books_df['Book-Title']))


    top3_recommendations = get_top3_recommendations(recs_pred)
    isbn_to_name = map_isbn_to_names()
    for uid, user_ratings in top3_recommendations.items():
        print(uid, [isbn_to_name[iid] for (iid, _) in user_ratings])


    # Training and testing based on split data set
    # train_set, test_set = data.split_ratings_train_test(ratings_df_param_densified)
    # algo.fit(train_set)
    # test_pred = algo.test(test_set)
    #
    # accuracy.rmse(test_pred)
