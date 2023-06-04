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
    slice_point = 0
    for arg in sys.argv:
        if '-s' in arg: slice_point = float(arg.split('=')[-1])
    
    for t in range(4):
        for oversampled in [0, 1]:
            X_data, y_data = load_data('validation')
            y_data = y_data[:, t]
            clfs = glob(f'models/*Classifier*t{t + 1}s{slice_point}o{oversampled}*')
            regs = sorted(glob(f'models/*Regressor*t{t + 1}s{slice_point}*o{oversampled}*'))

            clfs_names = [clf.split('/')[1].split('Classifier')[0] for clf in clfs]
            regs_names = [reg.split('/')[1].split('Regressor')[0] for reg in regs]

            clfs = [load(open(clf, 'rb')) for clf in clfs]
            regs = [load(open(reg, 'rb')) for reg in regs]

            clfs_results = [clf.predict_proba(X_data) for clf in clfs]
            clfs_results = [[result, np.argmax(result, axis = 1)] for result in clfs_results]
            regs_results = [reg.predict(X_data) for reg in regs]

            df = pd.DataFrame(columns = ['target', 'slice_point', 'oversampled',
                                         'clf', 'reg0', 'reg1', 'reg2',
                                         'reg0_is_post', 'reg1_is_post', 'reg2_is_post',
                                         'mae', 'amae'])
            
            for form in product([0, 1], # clf
                                [0, 3], # reg0
                                [1, 4], # reg1
                                [2, 5], # reg2
                                [0, 1], # class or prob reg0
                                [0, 1], # class or prob reg1
                                [0, 1]  # class or prob reg2
                            ):
                
                clf, reg0, reg1, reg2, m0, m1, m2 = form

                # class 0 prediction
                inx = clfs_results[clf][1] == 0
                y_obs = y_data[inx]
                if m0 == 0: y_hat = regs_results[reg0][inx]
                else: y_hat = regs_results[reg0][inx] * clfs_results[clf][0][inx, 0] + \
                              regs_results[reg1][inx] * clfs_results[clf][0][inx, 1] + \
                              regs_results[reg2][inx] * clfs_results[clf][0][inx, 2]
                
                # class 1 prediction
                inx = clfs_results[clf][1] == 1
                y_obs = np.hstack([y_obs, y_data[inx]])
                if m1 == 0: y_aux = regs_results[reg0][inx]
                else: y_aux = regs_results[reg0][inx] * clfs_results[clf][0][inx, 0] + \
                              regs_results[reg1][inx] * clfs_results[clf][0][inx, 1] + \
                              regs_results[reg2][inx] * clfs_results[clf][0][inx, 2]
                
                y_hat = np.hstack([y_hat, y_aux])
                
                # class 2 prediction
                inx = clfs_results[clf][1] == 2
                y_obs = np.hstack([y_obs, y_data[inx]])
                if m2 == 0: y_aux = regs_results[reg0][inx]
                else: y_aux = regs_results[reg0][inx] * clfs_results[clf][0][inx, 0] + \
                              regs_results[reg1][inx] * clfs_results[clf][0][inx, 1] + \
                              regs_results[reg2][inx] * clfs_results[clf][0][inx, 2]
                
                y_hat = np.hstack([y_hat, y_aux])

                final_mae = MAE(y_hat, y_obs)
                final_amae = AMAE(y_obs, y_hat, points = 1000, show = False)

                row = [
                    t + 1,
                    slice_point,
                    oversampled == 1,
                    clfs_names[clf],
                    regs_names[reg0],
                    regs_names[reg1],
                    regs_names[reg2],
                    m0 == 1,
                    m1 == 1,
                    m2 == 1,
                    final_mae,
                    final_amae
                ]

                df.loc[len(df)] = row

            if 'results' not in os.listdir(): os.mkdir('results')
            df.to_csv(f'results/t{t + 1}s{slice_point}o{oversampled}.csv', index = False)