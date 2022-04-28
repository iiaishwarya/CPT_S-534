from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score, mean_absolute_error

import numpy as np


def Decision_tree(X_train, y_train, X_test, y_test, min_samples_split):
    tree = DecisionTreeRegressor(
        min_samples_split=min_samples_split, max_depth=11, min_samples_leaf=1, max_features=None)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    train_rmse = np.sqrt(mse(y_train, y_train_pred))

    y_test_pred = tree.predict(X_test)
    test_rmse = np.sqrt(mse(y_test, y_test_pred))

    r2_train = r2_score(y_train, y_train_pred)

    r2_test = r2_score(y_test, y_test_pred)

    mae = (abs(y_test - y_test_pred)).mean()

    return train_rmse, test_rmse, r2_train, r2_test, y_train_pred, y_test_pred, mae
