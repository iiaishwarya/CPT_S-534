import imp
from operator import imod
from sklearn.model_selection import train_test_split

import preprocessing as pp
import Algorithms.DecisionTree as model_decision_tree
import Algorithms.NN as nn
import Algorithms.KNN_regressor as knnr
import Algorithms.LR


import numpy as np

X_train, y_train, X_test, y_test = pp.preprocessedData()

# Decision Tree

train_rmse, test_rmse, r2_train, r2_test, y_train_pred, y_test_pred, mae = model_decision_tree.Decision_tree(
    X_train, y_train, X_test, y_test, 2)

print("Results ---- Decision Tree")

print("Root Mean Squared Error for Train dataset is {}".format(train_rmse))
print("Root Mean Squared Error for Test  dataset is {}".format(test_rmse))
print("r2-score for Train Dataset is {}".format(r2_train))
print("r2-score for Test Dataset is {}".format(r2_test))
print("Mean Absolute Error for Test dataset is {}".format(mae))

print("Results ---- Decision Tree")


# KNN Regressor

train_rmse, test_rmse, r2_train, r2_test, y_train_pred, y_test_pred, mae = knnr.model_KNN_regressor(
    X_train, y_train, X_test, y_test)


# Neural Network

nn.modelNN(X_train, y_train, X_test, y_test)

# Linear Regression

train_rmse, test_rmse, r2_train, r2_test, y_train_pred, y_test_pred, mae = Algorithms.LR.model_linear(
    X_train, y_train, X_test, y_test)

print("Results ---- Linear Regression")

print("Root Mean Squared Error for Train dataset is {}".format(train_rmse))
print("Root Mean Squared Error for Test  dataset is {}".format(test_rmse))
print("r2-score for Train Dataset is {}".format(r2_train))
print("r2-score for Test Dataset is {}".format(r2_test))
print("Mean Absolute Error for Test dataset is {}".format(mae))

print("Results ---- Linear Regression")
