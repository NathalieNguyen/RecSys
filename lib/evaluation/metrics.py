from surprise import accuracy


def MAE(test_pred):
    return accuracy.mae(test_pred, verbose=False)


def RMSE(test_pred):
    return accuracy.rmse(test_pred, verbose=False)
