from lib.preprocessing import data_for_training as data
from surprise import KNNBasic
from lib.evaluation import metrics as metrics
from lib.evaluation import recommendations as recs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


if __name__ == '__main__':
    np.random.seed(0)

    stratify_df = data.filter_ratings_above_count_threshold(sample=20000)

    # Choose either densified ratings df or sample for building the train set
    train_set, test_set = train_test_split(stratify_df, stratify=stratify_df['User-ID'])

    IxU_train = csr_matrix(train_set.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0))
    IxU_test = csr_matrix(test_set.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0))

    print(IxU_train)
    print(IxU_test)

    # item_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    # item_knn.fit(IxU_train)

    # Training and predictions based on full data set with anti test set for building recommendations
    # full_train_set = data.load_full_ratings_train_set(ratings_df_param_densified)
    #
    # item_knn = KNNBasic(k=10, sim_options={'name': 'pearson_baseline', 'user_based': False})
    # item_knn.fit(full_train_set)
    #
    # anti_test_set = full_train_set.build_anti_testset()
    # recs_pred = item_knn.test(anti_test_set)
    #
    # print(recs.show_recommendations(recs_pred=recs_pred, user_id=276994))

    # Training and predictions based on split data set for evaluating metrics
    # item_knn.fit(IxU_train)
    # test_pred = item_knn.test(IxU_test)

    # print("Item-based collaborative filtering with KNN: Test set")
    # print('MAE: ', metrics.MAE(test_pred))
    # print('RMSE: ', metrics.RMSE(test_pred))
