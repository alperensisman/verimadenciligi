from sklearn import preprocessing
import numpy as np
from scipy import stats
# pip install scikit-learn
# pip install numpy
def minMaxNormalizasyon():
    X_train = np.array([[ 1., -1.,  2.],
    [ 2.,  0.,  0.],
    [ 0.,  1., -1.]])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    print(X_train_minmax)

def zScore():
    arr1 = [[20, 2, 7, 1, 34],
            [50, 12, 12, 34, 4]]
    arr2 = [[50, 12, 12, 34, 4], 
            [12, 11, 10, 34, 21]]
    print ("\nZ-score for arr1 : \n", stats.zscore(arr1))
    print ("\nZ-score for arr1 : \n", stats.zscore(arr1, axis = 1))


minMaxNormalizasyon()
zScore()