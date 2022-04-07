# Preprocessing
import preprocessing as pp
import pandas as pd

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Linear Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
def preprocess():
    print("Preprocessing data...")
    x_train, y_train, x_test, y_test = pp.preprocessedData()
    # DEBUG ONLY
    #print(len(x_train))
    #for index, row in x_train.iterrows():
    #    print(row["acousticness"])

    scaler = StandardScaler()
    scaler.fit(x_train)

    #x_train = scaler.transform(x_train)

    label_encoder = preprocessing.LabelEncoder()
    #encoded1 = lab_enc.fit_transform(x_train)
    encoded_y_train = label_encoder.fit_transform(y_train)
    encoded_y_test = label_encoder.fit_transform(y_test)

    print("Preprocessing data complete.")

    return x_train, encoded_y_train, x_test, encoded_y_test

def k_nearest_neighbors(x_train, y_train, x_test, y_test, k, test=False):
    print("Starting KNN algorithm...")
    if test == False:
        print("Starting KNN with k = " + str(k) + "...")
        # train
        n = KNeighborsClassifier(n_neighbors = k)
        n.fit(x_train, y_train)

        # test prediction
        pred_y = n.predict(x_test)
        print("Accuracy at K = " + str(k) + ": " + str(metrics.accuracy_score(y_test, pred_y)))
    else:
        print("Starting KNN testing loop...")
        for i in range(3, k):
            # train
            n = KNeighborsClassifier(n_neighbors = i)
            n.fit(x_train, y_train)

            # test prediction
            pred_y = n.predict(x_test)
            print("Accuracy at K = " + str(i) + ": " + str(metrics.accuracy_score(y_test, pred_y)))
    
    print("KNN algorithm complete.")

def Logistic_Regression(x_train, y_train, x_test, y_test):
    print("Starting Logistic Regression algorithm...")
    #lr = LogisticRegression(C=10, max_iter=100000, dual=False) #max_iter=1000)
    lr = LogisticRegression(solver='liblinear')
    lr.fit(x_train, y_train)
    prediction = lr.predict(x_test)
    print("Accuracy of linear regression: ", metrics.accuracy_score(y_test, prediction))
    print("Logistic Regression algorithm complete.")

x_train, y_train, x_test, y_test = preprocess()
k_nearest_neighbors(x_train, y_train, x_test, y_test, 40, True)
#Logistic_Regression(x_train, y_train, x_test, y_test)
