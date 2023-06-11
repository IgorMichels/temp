import sys
import os

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
    if 'models' not in os.listdir(): os.mkdir('models')
    X_train, y_train = load_data()
    t, slice_point, oversampled = 0, 0, 0
    for arg in sys.argv:
        if '-t' in arg: t = int(arg.split('=')[-1])
        if '-s' in arg: slice_point = float(arg.split('=')[-1])
        if '-o' in arg: oversampled = int(arg.split('=')[-1])

    if oversampled: X_train_aux, y_train_aux = oversampling(X_train, y_train[:, t])
    else: X_train_aux, y_train_aux = X_train, y_train[:, t]
    
    print(f'target {t + 1}\nslice: {slice_point}\n')
    y_train_classes = 1 * (y_train_aux <= slice_point) - 1 * (y_train_aux == 0) + 2 * (y_train_aux > slice_point)
    print('Classifiers')
    print('  Random Forest')
    clfFile = f'RandomForestClassifier - t{t + 1}s{slice_point}o{oversampled}.sav'
    if clfFile not in os.listdir('models/'):
        clf = RandomForestClassifier(n_estimators = 100, random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        clf.fit(X_train_aux, y_train_classes);
        dump(clf, open('models/' + clfFile, 'wb'))

    print('  Hist Gradient Boosting')
    clfFile = f'HistGradientBoostingClassifier - t{t + 1}s{slice_point}o{oversampled}.sav'
    if clfFile not in os.listdir('models/'):
        clf = HistGradientBoostingClassifier(random_state = 0, max_depth = 10, max_leaf_nodes = None, min_samples_leaf = 30)
        clf.fit(X_train_aux, y_train_classes);
        dump(clf, open('models/' + clfFile, 'wb'))

    print('  XGBoost')
    clfFile = f'XGBClassifier - t{t + 1}s{slice_point}o{oversampled}.sav'
    if clfFile not in os.listdir('models/'):
        clf = XGBClassifier(random_state = 0, max_depth = 10)
        clf.fit(X_train_aux, y_train_classes);
        dump(clf, open('models/' + clfFile, 'wb'))

    print('Regressors - class 0')
    indx = y_train_classes == 0
    print('  Random Forest')
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

    print('  XGBoost')
    regFile = f'XGBRegressor - t{t + 1}s{slice_point}c0o{oversampled}.sav'
    if regFile not in os.listdir('models/'):
        reg = XGBRegressor(n_estimators = 100, random_state = 0, max_depth = 10, n_jobs = -1)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))

    print('Regressors - class 1')
    indx = y_train_classes == 1
    print('  Random Forest')
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

    print('  XGBoost')
    regFile = f'XGBRegressor - t{t + 1}s{slice_point}c1o{oversampled}.sav'
    if regFile not in os.listdir('models/'):
        reg = XGBRegressor(n_estimators = 100, random_state = 0, max_depth = 10, n_jobs = -1)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))

    print('Regressors - class 2')
    indx = y_train_classes == 2
    print('  Random Forest')
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

    print('  XGBoost')
    regFile = f'XGBRegressor - t{t + 1}s{slice_point}c2o{oversampled}.sav'
    if regFile not in os.listdir('models/'):
        reg = XGBRegressor(n_estimators = 100, random_state = 0, max_depth = 10, n_jobs = -1)
        reg.fit(X_train_aux[indx, :], y_train_aux[indx]);
        dump(reg, open('models/' + regFile, 'wb'))