# -*- coding: utf-8 -*-
"""Untitled20.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hgfX_MHRU0EYutv3QbzRw0lmm8c4dnod
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
# figure size in inches
# %matplotlib inline

tracks_data = pd.read_csv('tracks.csv',index_col=0)
artists_data= pd.read_csv('artists.csv')

plt.figure(figsize=(16, 8))
sns.set(style="whitegrid")
corr = tracks_data.corr()
sns.heatmap(corr,annot=True)

plt.figure(figsize=(16, 8))
sns.set(style="whitegrid",palette='Blues_d')
x = tracks_data.groupby("artists")["popularity"].sum().sort_values(ascending=False).head(10)
ax = sns.barplot(x.index, x)
ax.set_title('Top Artists with Popularity')
ax.set_ylabel('Popularity')
ax.set_xlabel('Artists')
plt.xticks(rotation = 90)

plt.figure(figsize=(16, 8))
sns.set(style="whitegrid",palette='Blues_d')
x = tracks_data.groupby("name")["popularity"].mean().sort_values(ascending=False).head(10)
ax = sns.barplot(x.index, x)
ax.set_title('Top Tracks with Popularity')
ax.set_ylabel('Popularity')
ax.set_xlabel('Tracks')
plt.xticks(rotation = 90)