import numpy as np
import os

from sys import argv
from pickle import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

# global data
X_train = np.load('X_train0.npy')
for i in range(1, 10):
    aux = np.load(f'X_train{i}.npy')
    X_train = np.vstack([X_train, aux])

y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

def modeling(slice_point, target):
    global X_train
    global y_train
    global X_test
    global y_test

    # creating classes
    y_train_classes = 1 * (y_train <= slice_point) - 1 * (y_train == 0) + 2 * (y_train > slice_point)
    clfFile = f'HistGradientBoostingClassifier - t{target}s{slice_point}.sav'
    if clfFile not in os.listdir('models/'):
        clf  = HistGradientBoostingClassifier(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        clf.fit(X_train, y_train_classes[:, target - 1]);
        dump(clf, open('models/' + clfFile, 'wb'))

    clfFile = f'RandomForestClassifier - t{target}s{slice_point}.sav'
    if clfFile not in os.listdir('models/'):
        clf  = RandomForestClassifier(n_estimators = 10, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        clf.fit(X_train, y_train_classes[:, target - 1]);
        dump(clf, open('models/' + clfFile, 'wb'))
    
    # modelo HGB
    regFile = f'HistGradientBoostingRegressor - t{target}s{slice_point}c0.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes[:, target - 1] == 0
        reg0 = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg0.fit(X_train[indx, :], y_train[indx, target - 1]);
        dump(reg0, open('models/' + regFile, 'wb'))
    
    regFile = f'HistGradientBoostingRegressor - t{target}s{slice_point}c1.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes[:, target - 1] == 1
        reg1 = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg1.fit(X_train[indx, :], y_train[indx, target - 1]);
        dump(reg1, open('models/' + regFile, 'wb'))

    regFile = f'HistGradientBoostingRegressor - t{target}s{slice_point}c2.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes[:, target - 1] == 2
        reg2 = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg2.fit(X_train[indx, :], y_train[indx, target - 1]);
        dump(reg2, open('models/' + regFile, 'wb'))
    
    # modelo RF
    regFile = f'RandomForestRegressor - t{target}s{slice_point}c0.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes[:, target - 1] == 0
        reg0 = RandomForestRegressor(n_estimators = 10, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg0.fit(X_train[indx, :], y_train[indx, target - 1]);
        dump(reg0, open('models/' + regFile, 'wb'))

    regFile = f'RandomForestRegressor - t{target}s{slice_point}c1.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes[:, target - 1] == 1
        reg1 = RandomForestRegressor(n_estimators = 10, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg1.fit(X_train[indx, :], y_train[indx, target - 1]);
        dump(reg1, open('models/' + regFile, 'wb'))
    
    regFile = f'RandomForestRegressor - t{target}s{slice_point}c2.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes[:, target - 1] == 2
        reg2 = RandomForestRegressor(n_estimators = 10, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg2.fit(X_train[indx, :], y_train[indx, target - 1]);
        dump(reg2, open('models/' + regFile, 'wb'))

if __name__ == '__main__':
    _, target, slice_point = argv
    target = int(target.split('=')[1])
    slice_point = float(slice_point.split('=')[1])
    modeling(slice_point, target)