import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, InputLayer
from tensorflow.keras import optimizers, regularizers, metrics
alpha = 0.2

test = 'Dataset/data_test.csv'
train = 'Dataset/data_train.csv'
valid = 'Dataset/data_val.csv'

data_test = pd.read_csv(test)
data_train = pd.read_csv(train)
data_val = pd.read_csv(valid)

X_train = minmax_scale(data_train.iloc[:, 0:100], axis=0)
y_train = minmax_scale(data_train.iloc[:, -1], axis=0)


X_test = minmax_scale(data_test.iloc[:, 0:100], axis=0)
y_test = minmax_scale(data_test.iloc[:, -1], axis=0)

X_val = minmax_scale(data_val.iloc[:, 0:100], axis=0)
y_val = minmax_scale(data_val.iloc[:, -1], axis=0)

model = Sequential()

input_dim = (100,)

model.add(InputLayer(input_shape=(X_train.shape[1],)))
model.add(Dense(500, activation='relu', input_shape=input_dim))
model.add(Dense(200, activation='relu',
          activity_regularizer=regularizers.l2(alpha)))
model.add(Dense(100, activation='relu',
          activity_regularizer=regularizers.l2(alpha)))
model.add(Dense(10, activation='relu',
          activity_regularizer=regularizers.l2(alpha)))
model.add(Dense(1, activation='sigmoid'))

adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9,
                       beta_2=0.999,
                       epsilon=1e-07,)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[
                  metrics.MeanSquaredError(),
                  metrics.RootMeanSquaredError(),
                  metrics.MeanAbsoluteError()
              ])
train_history = []
valid_history = []
test_history = []

for i in range(10):
    history = model.fit(X_train, y_train, epochs=1,
                        batch_size=32, shuffle=True)
    train_history.append(list(history.history.values())[0][0])
    score_valid = model.evaluate(X_val, y_val, verbose=1)
    valid_history.append(score_valid)
    score_test = model.evaluate(X_test, y_test, verbose=1)
    test_history.append(score_test)
