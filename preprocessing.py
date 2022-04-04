import pandas as pd
import numpy as np
import datetime

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split


tracks = pd.read_csv('/content/tracks.csv')
artists = pd.read_csv('/content/artists.csv')
tracks.head()
artists.head()

tracks.describe()

# Data Cleaning

# millisconds to minutes
tracks['duration_min'] = tracks['duration_ms']/60000
tracks['duration_min'] = tracks['duration_min'].round(2)


# # Remove bracket from artist names
# tracks["artists"]=tracks["artists"].str.replace("[", "")
# tracks["artists"]=tracks["artists"].str.replace("]", "")
# tracks["artists"]=tracks["artists"].str.replace("'", "")
tracks['release_date'] = pd.to_datetime(
    tracks['release_date'], infer_datetime_format=True)

tracks['release_date']
tracks['year'] = tracks['release_date'].dt.year

# # #Add Song decade column in the dataset
tracks['Song Decade'] = None

tracks.loc[(tracks['year'] >= 1920) & (
    tracks['year'] < 1930), 'Song Decade'] = '1920s'
tracks.loc[(tracks['year'] >= 1930) & (
    tracks['year'] < 1940), 'Song Decade'] = '1930s'
tracks.loc[(tracks['year'] >= 1940) & (
    tracks['year'] < 1950), 'Song Decade'] = '1940s'
tracks.loc[(tracks['year'] >= 1950) & (
    tracks['year'] < 1960), 'Song Decade'] = '1950s'
tracks.loc[(tracks['year'] >= 1960) & (
    tracks['year'] < 1970), 'Song Decade'] = '1960s'
tracks.loc[(tracks['year'] >= 1970) & (
    tracks['year'] < 1980), 'Song Decade'] = '1970s'
tracks.loc[(tracks['year'] >= 1980) & (
    tracks['year'] < 1990), 'Song Decade'] = '1980s'
tracks.loc[(tracks['year'] >= 1990) & (
    tracks['year'] < 2000), 'Song Decade'] = '1990s'
tracks.loc[(tracks['year'] >= 2000) & (
    tracks['year'] < 2010), 'Song Decade'] = '2000s'
tracks.loc[(tracks['year'] >= 2010) & (
    tracks['year'] < 2020), 'Song Decade'] = '2010s'
tracks.loc[(tracks['year'] >= 2020) & (
    tracks['year'] < 2030), 'Song Decade'] = '2020s'

tracks.head()


tracks.duplicated().any().sum()

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'duration_ms',
                 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration_min', 'explicit', 'key', 'mode', 'year', 'artists']

X, y = tracks[feature_names], tracks['popularity']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train.head()


# Data transformation

class Artists:
    def __init__(self, MinCnt=3.0, MaxCnt=600.0):
        self.MinCnt = MinCnt
        self.MaxCnt = MaxCnt
        self.artists_df = None

    def fit(self, X, y):
        self.artists_df = y.groupby(X.artists).agg(['mean', 'count'])
        self.artists_df.loc['unknown'] = [y.mean(), 1]
        self.artists_df.loc[self.artists_df['count']
                            <= self.MinCnt, 'mean'] = y.mean()
        self.artists_df.loc[self.artists_df['count']
                            >= self.MaxCnt, 'mean'] = 0
        return self

    def transform(self, X, y=None):
        X['artists'] = np.where(X['artists'].isin(
            self.artists_df.index), X['artists'], 'unknown')
        X['artists'] = X['artists'].map(self.artists_df['mean'])
        return X


def instrumental(X):
    X['instrumentalness'] = list(
        map((lambda x: 1 if x < 0.1 else (3 if x > 0.95 else 2)), X.instrumentalness))


def transform_tempo(X):
    X.loc[X['tempo'] == 0, 'tempo'] = X.loc[X['tempo'] > 0, 'tempo'].mean()
    return X


# Apply Aritists class on train and test seperatly
artists_transformer = Artists(MinCnt=2)
X_train = artists_transformer.fit(X_train, y_train).transform(X_train, y_train)
X_test = artists_transformer.transform(X_test, y_test)

# Apply Instrumental Criteria on train & test seperately
instrumentalness_tranformer = FunctionTransformer(instrumental)
instrumentalness_tranformer.transform(X_train)
instrumentalness_tranformer.transform(X_test)

# Apply Tempo Transformer class on Train & Test seperately
X_train = transform_tempo(X_train)
X_test = transform_tempo(X_test)

# One Hot Encoding

ohe = OneHotEncoder(categories='auto', drop='first')

# Train
feature_arr = ohe.fit_transform(X_train[['instrumentalness', 'key']]).toarray()
columns_key = ['key_'+str(i) for i in list(set(X_train['key'].values))[1:]]
instrumentalness_key = [
    'ins_'+str(i) for i in list(set(X_train['instrumentalness'].values))[1:]]
feature_labels = columns_key + instrumentalness_key
feature_labels = np.concatenate((feature_labels), axis=None)
features = pd.DataFrame(
    feature_arr, columns=feature_labels, index=X_train.index)
X_train = pd.concat([X_train, features], axis=1).drop(
    ['key', 'instrumentalness'], axis=1)

# Test
feature_arr = ohe.fit_transform(X_test[['instrumentalness', 'key']]).toarray()
columns_key = ['key_'+str(i) for i in list(set(X_test['key'].values))[1:]]
instrumentalness_key = [
    'ins_'+str(i) for i in list(set(X_test['instrumentalness'].values))[1:]]
feature_labels = columns_key + instrumentalness_key
feature_labels = np.concatenate((feature_labels), axis=None)
features = pd.DataFrame(
    feature_arr, columns=feature_labels, index=X_test.index)
X_test = pd.concat([X_test, features], axis=1).drop(
    ['key', 'instrumentalness'], axis=1)

# MinMaxScaler

scaler = MinMaxScaler()
cols = ['artists', 'duration_ms', 'loudness', 'tempo']
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.fit_transform(X_test[cols])

# Divide the popularity by 100
y_train = y_train / 100
y_test = y_test / 100


def getData():
    return X_train, y_train, X_test, y_test
