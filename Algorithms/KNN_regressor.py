from sklearn.neighbors import KNeighborsRegressor
import preprocessing as pp
import numpy as np

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score, mean_absolute_error


def model_KNN_regressor(X_train, y_train, X_test, y_test):
    tree = KNeighborsRegressor(n_neighbors=6)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    train_rmse = np.sqrt(mse(y_train, y_train_pred))

    y_test_pred = tree.predict(X_test)
    test_rmse = np.sqrt(mse(y_test, y_test_pred))

    r2_train = r2_score(y_train, y_train_pred)

    r2_test = r2_score(y_test, y_test_pred)

    mae = (abs(y_test - y_test_pred)).mean()

    return train_rmse, test_rmse, r2_train, r2_test, y_train_pred, y_test_pred, mae
