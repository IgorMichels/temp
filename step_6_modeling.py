import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from utils import oversampling
from utils import load_data
from joblib import dump

if __name__ == '__main__':
    if 'final_models' not in os.listdir(): os.mkdir('final_models')
    line = 0
    for arg in sys.argv:
        if '-l' in arg: line = int(arg.split('=')[-1])

    df = pd.read_csv('final.csv')
    t, slice_point, oversampled, model_name, model = df.loc[line].values
    X_train, y_train = load_data()
    X_train_aux, y_train_aux = load_data('validation')
    X_train = np.vstack([X_train, X_train_aux])
    y_train = np.hstack([y_train, y_train_aux])
    del X_train_aux, y_train_aux

    y_train = y_train[:, t - 1]
    if oversampled: X_train_aux, y_train_aux = oversampling(X_train, y_train)
    else: X_train_aux, y_train_aux = X_train, y_train

    if model == 'clf':
        print('Classifier')
        y_train_classes = 1 * (y_train_aux <= slice_point) - 1 * (y_train_aux == 0) + 2 * (y_train_aux > slice_point)
        clf_file = f'{model_name}Classifier - t{t + 1}s{slice_point}o{oversampled}.sav'
        if clf_file not in os.listdir('final_models/'):
            if model_name == 'RandomForest': clf = RandomForestClassifier(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
            elif model_name == 'XGB': clf = XGBClassifier(random_state = 0, max_depth = 10)
            else: clf = HistGradientBoostingClassifier(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
            clf.fit(X_train_aux, y_train_classes);
            dump(clf, open('final_models/' + clf_file, 'wb'))
    elif model == 'reg1':
        print('Regression class 1')
        reg_file = f'{model_name}Regressor - t{t + 1}s{slice_point}c1o{oversampled}.sav'
        if reg_file not in os.listdir('final_models/'):
            indx = (y_train_aux > 0) * (y_train_aux <= slice_point)
            if model_name == 'RandomForest': reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
            elif model_name == 'XGB': reg = XGBRegressor(n_estimators = 100, random_state = 0, max_depth = 10, n_jobs = -1)
            else: reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
            reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
            dump(reg, open('final_models/' + reg_file, 'wb'))
    elif model == 'reg2':
        print('Regression class 2')
        reg_file = f'{model_name}Regressor - t{t + 1}s{slice_point}c2o{oversampled}.sav'
        if reg_file not in os.listdir('final_models/'):
            indx = y_train_aux > slice_point
            if model_name == 'RandomForest': reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
            elif model_name == 'XGB': reg = XGBRegressor(n_estimators = 100, random_state = 0, max_depth = 10, n_jobs = -1)
            else: reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
            reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
            dump(reg, open('final_models/' + reg_file, 'wb'))
