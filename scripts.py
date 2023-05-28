import os
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

def load_data():
    for i in range(20):
        if i == 0:
            X_train = np.load('X_train0.npy')
            y_train = np.load('y_train0.npy')
        else:
            X_train = np.vstack([X_train, np.load(f'X_train{i}.npy')])
            y_train = np.vstack([y_train, np.load(f'y_train{i}.npy')])

    return X_train, y_train

def oversampling(X, y, step = 2):
    X_array = X.copy()
    y_array = y.copy()
    for i in range(step, 101, step):
        X_array = np.vstack([X_array, X[y >= i]])
        y_array = np.hstack([y_array, y[y >= i]])

    return X_array, y_array

def train(X_train, y_train, t, slice_point):
    X_train_aux, y_train_aux = oversampling(X_train, y_train[:, t])
    print(f'target {t + 1}\nslice: {slice_point}\n')
    y_train_classes = 1 * (y_train_aux <= slice_point) - 1 * (y_train_aux == 0) + 2 * (y_train_aux > slice_point)
    print('Classifiers')
    print('  Random Forest')
    clfFile = f'RandomForestClassifier - t{t + 1}s{slice_point}.sav'
    if clfFile in os.listdir('models/'):
        clf = load(open('models/' + clfFile, 'rb'))
    else:
        clf  = RandomForestClassifier(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        clf.fit(X_train_aux, y_train_classes);
        dump(clf, open('models/' + clfFile, 'wb'))

    print('  Hist Gradient Boosting')
    clfFile = f'HistGradientBoostingClassifier - t{t + 1}s{slice_point}.sav'
    if clfFile in os.listdir('models/'):
        clf = load(open('models/' + clfFile, 'rb'))
    else:
        clf  = HistGradientBoostingClassifier(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        clf.fit(X_train_aux, y_train_classes);
        dump(clf, open('models/' + clfFile, 'wb'))

    print('Regressors - class 1')
    print('  Random Forest')
    regFile = f'RandomForestRegressor - t{t + 1}s{slice_point}c0.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes == 0
        reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))

    print('  Hist Gradient Boosting')
    regFile = f'HistGradientBoostingRegressor - t{t + 1}s{slice_point}c0.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes == 0
        reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))

    print('Regressors - class 2')
    print('  Random Forest')
    regFile = f'RandomForestRegressor - t{t + 1}s{slice_point}c1.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes == 1
        reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))
    
    print('  Hist Gradient Boosting')
    regFile = f'HistGradientBoostingRegressor - t{t + 1}s{slice_point}c1.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes == 1
        reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))

    print('Regressors - class 3')
    print('  Random Forest')
    regFile = f'RandomForestRegressor - t{t + 1}s{slice_point}c2.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes == 2
        reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))
    
    print('  Hist Gradient Boosting')
    regFile = f'HistGradientBoostingRegressor - t{t + 1}s{slice_point}c2.sav'
    if regFile not in os.listdir('models/'):
        indx = y_train_classes == 2
        reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))