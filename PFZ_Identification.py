import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn import preprocessing
np.random.seed(4)

td = pd.read_csv('sample_dataset.csv')
td.head(5)

'''
Sea Surface Temperature, Chlorophyll, 
Relative Humidity, Sea Level Pressure, Air Temperature, Total Cloudiness and Total Fish catch data.

'''
td['Label'] = td[col[-1]].map({'PFZ':1,'NPFZ':0}) # maping

X = td[[col[2],col[3],col[4],col[5],col[6],col[7],col[8]]]
y = td[col[-1]]

X_preprocess = preprocessing.scale(X)
X_normalized = preprocessing.normalize(X_preprocess, norm='l2')



# # Feature Selection ---------> 
# # 1. Univariate Selection
# Statistical tests can be used to select those features that have the strongest relationship 
# with the output variable.
# The example below uses the chi squared (chi^2) statistical test for non-negative features 
# to select 4 of the best features
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=4) # Select best 4 best feature
fit = test.fit(np.abs(X_normalized), y)

# summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

val=fit.scores_[0:7]
print(np.where(val==np.max(val)))



# the scores for each attribute and the 4 attributes chosen (those with the highest scores
# # 2. Recursive Feature Elimination
# The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on 
# those attributes that remain.
# It uses the model accuracy to identify which attributes (and combination of attributes) 
# contribute the most to predicting 
# the target attribute.
# Feature Extraction with RFE

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# feature extraction
model = LogisticRegression()
rfe = RFE(model, 4)
fit = rfe.fit(X_normalized, y)
print("Num Features: %d", fit.n_features_)
print("Selected Features: %s", fit.support_)
print("Feature Ranking: %s",fit.ranking_)


# # 3. Principal Component Analysis
#Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form.

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
fit = pca.fit(X_normalized)
# summarize components
print("Explained Variance: %s",fit.explained_variance_ratio_)
print(fit.components_)


print(pd.DataFrame(pca.components_,columns=X.columns,index = ['PC-1','PC-2','PC-3','PC-4']))

# pca.components_ outputs an array of [n_components, n_features], so to get how components are linearly related
#with the different features


# # 4. Feature Importance

# Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.

from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
np.random.seed(1)
model = ExtraTreesClassifier()
model.fit(X_normalized, y)
print(model.feature_importances_)


# # Feature Selection

feature = td[[col[5],col[6],col[7],col[8]]]
feature = preprocessing.scale(feature)
feature = preprocessing.normalize(feature)
label   = np.array(y)

import keras
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dropout
from time import time
from keras import losses

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss=losses.mean_squared_error, optimizer='sgd', metrics=['accuracy'])
model.fit(feature,dummy_y,batch_size=32, epochs=1000)




