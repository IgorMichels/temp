import os
import sys
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from scripts import load_data, oversampling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from joblib import dump

if __name__ == '__main__':
    if 'refined_models' not in os.listdir(): os.mkdir('refined_models')
    X_train, y_train = load_data()
    t = 0
    slice_point = 0
    oversampled = 0
    for arg in sys.argv:
        if '-t' in arg: t = int(arg.split('=')[-1])
        if '-s' in arg: slice_point = int(arg.split('=')[-1])
        if '-o' in arg: oversampled = int(arg.split('=')[-1])

    y_train = y_train[:, t]
    if oversampled: X_train_aux, y_train_aux = oversampling(X_train, y_train)
    else: X_train_aux, y_train_aux = X_train, y_train

    df = pd.read_csv('retrain.csv')
    df = df[((df['target'] == t + 1) & (df['oversampled'] == oversampled))]
    slice_point = sorted(list(set(df['slice_point'])))[slice_point]
    df = df[df['slice_point'] == slice_point].reset_index(drop = True)

    for i in df.index:
        aux = df.loc[i]
        if aux['model'] == 'clf':
            print('Classifier')
            y_train_classes = 1 * (y_train_aux <= slice_point) - 1 * (y_train_aux == 0) + 2 * (y_train_aux > slice_point)
            clf_file = f"{aux['model_name']}Classifier - t{t + 1}s{slice_point}o{oversampled}.sav"
            if clf_file not in os.listdir('refined_models/'):
                if aux['model_name'] == 'RandomForest': clf = RandomForestClassifier(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
                elif aux['model_name'] == 'XGB': clf = XGBClassifier(random_state = 0, max_depth = 10)
                else: clf = HistGradientBoostingClassifier(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
                clf.fit(X_train_aux, y_train_classes);
                dump(clf, open('refined_models/' + clf_file, 'wb'))
        elif aux['model'] == 'reg1':
            print('Regression class 1')
            reg_file = f"{aux['model_name']}Regressor - t{t + 1}s{slice_point}c1o{oversampled}.sav"
            if reg_file not in os.listdir('refined_models/'):
                indx = y_train_classes == 1
                if aux['model_name'] == 'RandomForest': reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
                elif aux['model_name'] == 'XGB': reg = XGBRegressor(n_estimators = 100, random_state = 0, max_depth = 10, n_jobs = -1)
                else: reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
                reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
                dump(reg, open('refined_models/' + reg_file, 'wb'))
        elif aux['model'] == 'reg2':
            print('Regression class 2')
            reg_file = f"{aux['model_name']}Regressor - t{t + 1}s{slice_point}c2o{oversampled}.sav"
            if reg_file not in os.listdir('refined_models/'):
                indx = y_train_classes == 2
                if aux['model_name'] == 'RandomForest': reg = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
                elif aux['model_name'] == 'XGB': reg = XGBRegressor(n_estimators = 100, random_state = 0, max_depth = 10, n_jobs = -1)
                else: reg = HistGradientBoostingRegressor(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
                reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
                dump(reg, open('refined_models/' + reg_file, 'wb'))
