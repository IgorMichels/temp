import os
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from glob import glob

def load_data(folder = 'train'):
    for i, file in enumerate(sorted(glob(f'{folder}/X*'))):
        if i == 0: X_data = np.load(file)
        else: X_data = np.vstack([X_data, np.load(file)])
        
    for i, file in enumerate(sorted(glob(f'{folder}/y*'))):
        if i == 0: y_data = np.load(file)
        else: y_data = np.vstack([y_data, np.load(file)])

    return X_data, y_data

def oversampling(X, y, step = 2):
    X_array = X.copy()
    y_array = y.copy()
    for i in range(step, 101, step):
        X_array = np.vstack([X_array, X[y >= i]])
        y_array = np.hstack([y_array, y[y >= i]])

    return X_array, y_array

def train(X_train, y_train, t, slice_point, oversampled = True):
    if oversampled: X_train_aux, y_train_aux = oversampling(X_train, y_train[:, t])
    else: X_train_aux, y_train_aux = X_train, y_train[:, t]
    
    print(f'target {t + 1}\nslice: {slice_point}\n')
    y_train_classes = 1 * (y_train_aux <= slice_point) - 1 * (y_train_aux == 0) + 2 * (y_train_aux > slice_point)
    print('Classifiers')
    print('  Random Forest')
    clfFile = f'RandomForestClassifier - t{t + 1}s{slice_point}o{oversampled}.sav'
    if clfFile not in os.listdir('models/'):
        clf  = RandomForestClassifier(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        clf.fit(X_train_aux, y_train_classes);
        dump(clf, open('models/' + clfFile, 'wb'))

    print('  Hist Gradient Boosting')
    clfFile = f'HistGradientBoostingClassifier - t{t + 1}s{slice_point}o{oversampled}.sav'
    if clfFile not in os.listdir('models/'):
        clf  = HistGradientBoostingClassifier(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        clf.fit(X_train_aux, y_train_classes);
        dump(clf, open('models/' + clfFile, 'wb'))

    print('Regressors - class 0')
    print('  Random Forest')
    indx = y_train_classes == 0
    regFile = f'RandomForestRegressor - t{t + 1}s{slice_point}c0o{oversampled}.sav'
    if regFile not in os.listdir('models/'):
        reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))

    print('  Hist Gradient Boosting')
    regFile = f'HistGradientBoostingRegressor - t{t + 1}s{slice_point}c0o{oversampled}.sav'
    if regFile not in os.listdir('models/'):
        reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))

    print('Regressors - class 1')
    print('  Random Forest')
    indx = y_train_classes == 1
    regFile = f'RandomForestRegressor - t{t + 1}s{slice_point}c1o{oversampled}.sav'
    if regFile not in os.listdir('models/'):
        reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))
    
    print('  Hist Gradient Boosting')
    regFile = f'HistGradientBoostingRegressor - t{t + 1}s{slice_point}c1o{oversampled}.sav'
    if regFile not in os.listdir('models/'):
        reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))

    print('Regressors - class 2')
    print('  Random Forest')
    indx = y_train_classes == 2
    regFile = f'RandomForestRegressor - t{t + 1}s{slice_point}c2o{oversampled}.sav'
    if regFile not in os.listdir('models/'):
        reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))
    
    print('  Hist Gradient Boosting')
    regFile = f'HistGradientBoostingRegressor - t{t + 1}s{slice_point}c2o{oversampled}.sav'
    if regFile not in os.listdir('models/'):
        reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))