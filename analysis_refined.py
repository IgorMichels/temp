from itertools import product
from scripts import load_data
from utils import MAE, AMAE
from joblib import load
from glob import glob

import pandas as pd
import numpy as np
import sys
import os

if __name__ == '__main__':
    if 'results_refined' not in os.listdir(): os.mkdir('results_refined')
    for arg in sys.argv:
        if '-s' in arg: slice_point = int(arg.split('=')[-1])

    for target, oversampled in product([1, 2, 3, 4], [0, 1]):
        df = pd.read_csv('retrain.csv')
        df = df[((df['target'] == target) & (df['oversampled'] == oversampled))]
        slices = df['slice_point'].values
        slices = np.unique(slices)
        df = df[df['slice_point'] == slices[slice_point]].reset_index(drop = True)
        clfs, clfs_name = list(), list()
        reg1, reg2 = list(), list()
        reg1_name, reg2_name = list(), list()
        for i in df.index:
            filename = df.loc[i, 'model_name']
            if df.loc[i, 'model'] == 'clf':
                clfs_name.append(filename)
                filename += f'Classifier - t{target}s{slices[slice_point]}o{oversampled}.sav'
                clfs.append('refined_models/' + filename)
            
            if df.loc[i, 'model'] == 'reg1':
                reg1_name.append(filename)
                filename += f'Regressor - t{target}s{slices[slice_point]}c1o{oversampled}.sav'
                reg1.append('refined_models/' + filename)
            
            if df.loc[i, 'model'] == 'reg2':
                reg2_name.append(filename)
                filename += f'Regressor - t{target}s{slices[slice_point]}c2o{oversampled}.sav'
                reg2.append('refined_models/' + filename)
        
        X_data, y_data = load_data('validation')
        y_data = y_data[:, target - 1]
        clfs = [load(open(clf, 'rb')) for clf in clfs]
        reg1 = [load(open(reg, 'rb')) for reg in reg1]
        reg2 = [load(open(reg, 'rb')) for reg in reg2]

        clfs_results = [clf.predict_proba(X_data) for clf in clfs]
        clfs_results = [[result, np.argmax(result, axis = 1)] for result in clfs_results]
        reg1_results = [reg.predict(X_data) for reg in reg1]
        reg2_results = [reg.predict(X_data) for reg in reg2]
        
        df = pd.DataFrame(columns = ['target', 'slice_point', 'oversampled',
                                    'clf', 'reg1', 'reg2',
                                    'reg0_is_post', 'reg1_is_post', 'reg2_is_post',
                                    'mae', 'amae'])
        
        for clf, reg1, reg2, m0, m1, m2 in product(range(len(clfs)),
                                                range(len(reg1)),
                                                range(len(reg2)),
                                                [0, 1],
                                                [0, 1],
                                                [0, 1]):
            
            # class 0 prediction
            inx = clfs_results[clf][1] == 0
            y_obs = y_data[inx]
            if m0 == 0: y_hat = 0 * y_data[inx]
            else: y_hat = 0                       * clfs_results[clf][0][inx, 0] + \
                          reg1_results[reg1][inx] * clfs_results[clf][0][inx, 1] + \
                          reg2_results[reg2][inx] * clfs_results[clf][0][inx, 2]
            
            # class 1 prediction
            inx = clfs_results[clf][1] == 1
            y_obs = np.hstack([y_obs, y_data[inx]])
            if m1 == 0: y_aux = reg1_results[reg1][inx]
            else: y_aux = 0                       * clfs_results[clf][0][inx, 0] + \
                          reg1_results[reg1][inx] * clfs_results[clf][0][inx, 1] + \
                          reg2_results[reg2][inx] * clfs_results[clf][0][inx, 2]
            
            y_hat = np.hstack([y_hat, y_aux])
            
            # class 2 prediction
            inx = clfs_results[clf][1] == 2
            y_obs = np.hstack([y_obs, y_data[inx]])
            if m2 == 0: y_aux = reg2_results[reg2][inx]
            else: y_aux = 0                       * clfs_results[clf][0][inx, 0] + \
                          reg1_results[reg1][inx] * clfs_results[clf][0][inx, 1] + \
                          reg2_results[reg2][inx] * clfs_results[clf][0][inx, 2]
            
            y_hat = np.hstack([y_hat, y_aux])

            final_mae = MAE(y_hat, y_obs)
            final_amae = AMAE(y_obs, y_hat, points = 1000)

            row = [
                target,
                slices[slice_point],
                oversampled == 1,
                clfs_name[clf],
                reg1_name[reg1],
                reg2_name[reg2],
                m0 == 1,
                m1 == 1,
                m2 == 1,
                final_mae,
                final_amae
            ]

            df.loc[len(df)] = row

        df.to_csv(f'results_refined/t{target}s{slice_point}o{oversampled}.csv', index = False)