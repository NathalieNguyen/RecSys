from lib.preprocessing import data_for_training as data
from surprise import SVD
from lib.evaluation import metrics as metrics
from lib.evaluation import recommendations as recs
import numpy as np

if __name__ == '__main__':
    np.random.seed(0)

    # Choose either densified ratings df or sample for building the train set
    ratings_df_param_sample = data.match_uid_and_isbn().sample(n=10000)
    ratings_df_param_densified = data.densify_ratings_df()

    # Training and predictions based on full data set with anti test set for building recommendations
    # full_train_set = data.load_full_ratings_train_set(ratings_df_param_densified)

    svd = SVD()
    svd.fit(full_train_set)

    anti_test_set = full_train_set.build_anti_testset()
    recs_pred = svd.test(anti_test_set)

    # print(recs.show_recommendations(262459))

    # Training and predictions based on split data set for evaluating metrics
    train_set, test_set = data.load_split_ratings_train_test(ratings_df_param_densified)
    svd.fit(train_set)
    test_pred = svd.test(test_set)

    print("Matrix factorixation with SVD: Test set")
    print('MAE: ', metrics.MAE(test_pred))
    print('RMSE: ', metrics.RMSE(test_pred))