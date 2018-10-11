from lib.preprocessing import data_for_training as data
from surprise import KNNBasic
from lib.evaluation import metrics as metrics
from lib.evaluation import recommendations as recs
import numpy as np

if __name__ == '__main__':
    np.random.seed(0)

    # Choose either densified ratings df or sample for building the train set
    ratings_df_param_sample = data.match_uid_and_isbn().sample(n=10000)
    ratings_df_param_densified = data.densify_ratings_df()

    # Training and predictions based on full data set with anti test set for building recommendations
    full_train_set = data.load_full_ratings_train_set(ratings_df_param_densified)

    item_knn = KNNBasic(k=10, sim_options={'name': 'pearson_baseline', 'user_based': False})
    item_knn.fit(full_train_set)

    anti_test_set = full_train_set.build_anti_testset()
    recs_pred = item_knn.test(anti_test_set)

    print(recs.show_recommendations(recs_pred=recs_pred, user_id=262459))

    # Training and predictions based on split data set for evaluating metrics
    train_set, test_set = data.load_split_ratings_train_test(ratings_df_param_densified)
    item_knn.fit(train_set)
    test_pred = item_knn.test(test_set)

    print("Item-based collaborative filtering with KNN: Test set")
    print('MAE: ', metrics.MAE(test_pred))
    print('RMSE: ', metrics.RMSE(test_pred))
