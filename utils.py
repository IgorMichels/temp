import numpy as np
from glob import glob

def MAE(y_pred, y_obs):
    return np.mean(np.abs(y_pred - y_obs))

def AMAE(y_obs, y_pred, points = 1000):
    limits = np.linspace(0, 100, points)
    dx = limits[1] - limits[0]
    maes = list()
    for x in limits:
        inx = y_obs >= x
        maes.append(MAE(y_pred[inx], y_obs[inx]))

    return np.sum(maes) * dx

def load_data(folder = 'train'):
    for i, file in enumerate(sorted(glob(f'data/{folder}/X*'))):
        if i == 0: X_data = np.load(file)
        else: X_data = np.vstack([X_data, np.load(file)])
        
    for i, file in enumerate(sorted(glob(f'data/{folder}/y*'))):
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
