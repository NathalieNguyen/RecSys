from lib.preprocessing import ratings as ratings
from lib.preprocessing import data_for_training as data
from surprise import KNNWithMeans
from surprise import accuracy


if __name__ == '__main__':
    # Choose either densified ratings df or sample
    # train_set, test_set = data.split_ratings_train_test(ratings.load_raw_df().sample(n=10000))
    train_set, test_set = data.split_ratings_train_test(data.densify_ratings_df())

    # Use user_based true/false to switch between user-based or item-based collaborative filtering
    algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
    algo.fit(train_set)

    # Train the algorithm on the train set and predict ratings for the test set
    predictions = algo.test(test_set)

    # Compute RMSE
    accuracy.rmse(predictions)
