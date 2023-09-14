import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel

import pickle

# load dataset
data = pd.read_csv('./BTC_sum_plus_nn_features.csv')

# In practice, feature selection should be done after data pre-processing,
# so ideally, all the categorical variables are encoded into numbers,
# and then you can assess how deterministic they are of the target

# here for simplicity I will use only numerical variables
# select numerical columns:

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]

print(data.shape)

# separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['dv1_realized_volatility'], axis=1),
    data['dv1_realized_volatility'],
    test_size=0.3,
    random_state=0)

print(X_train.shape, X_test.shape)

# X_train의 누락된 값을 이전 값으로 채우기
X_train = X_train.fillna(method='ffill')

# X_test의 누락된 값을 이전 값으로 채우기
X_test = X_test.fillna(method='ffill')

# the features in the house dataset are in different scales
# so we train a scaler to scale them

scaler = StandardScaler()
scaler.fit(X_train)

# we train a Linear regression model and select
# features with higher coefficients.

# the LinearRegression object from sklearn is a non-regularised
# linear method. It fits by matrix multiplication and not 
# gradient descent.

# therefore we don't need to specify penalty and other parameters

sel_ = SelectFromModel(LinearRegression())

sel_.fit(scaler.transform(X_train), y_train)

# let's count the number of variables selected
selected_feat = X_train.columns[(sel_.get_support())]

# 저장할 파일 이름
file_name = "selected_feat.pkl"

# selected_feat 변수를 pkl 파일로 저장
with open(file_name, 'wb') as file:
    pickle.dump(selected_feat, file)

print(f"{file_name} 파일이 저장되었습니다.")

print(len(selected_feat))


# and now, let's compare the  amount of selected features
# with the amount of features which coefficient is above the
# mean coefficient, to make sure we understand the output of
# sklearn

print('total features: {}'.format((X_train.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

print(
    'features with coefficients greater than the mean coefficient: {}'.format(
        np.sum(
            np.abs(sel_.estimator_.coef_) > np.abs(
                sel_.estimator_.coef_).mean())))