import pandas as pd
import numpy as np

from itertools import product
from utils import load_data
from utils import MAE, AMAE
from joblib import load
from glob import glob

if __name__ == '__main__':
    for target, oversampled in product([1, 2, 3, 4], [False, True]):
        clf = load(open(glob(f'final_models/*Classifier*t{target}*o{oversampled}*')[0], 'rb'))
        reg1 = load(open(glob(f'final_models/*Regressor*t{target}*c1o{oversampled}*')[0], 'rb'))
        reg2 = load(open(glob(f'final_models/*Regressor*t{target}*c2o{oversampled}*')[0], 'rb'))

        clf_name = glob(f'final_models/*Classifier*t{target}*o{oversampled}*')[0].split('/')[1].split('Classifier')[0]
        reg1_name = glob(f'final_models/*Regressor*t{target}*c1o{oversampled}*')[0].split('/')[1].split('Regressor')[0]
        reg2_name = glob(f'final_models/*Regressor*t{target}*c2o{oversampled}*')[0].split('/')[1].split('Regressor')[0]

        X_data, y_data = load_data('test')
        y_data = y_data[:, target - 1]

        clf_results = clf.predict_proba(X_data)
        clf_results = [clf_results, np.argmax(clf_results, axis = 1)]
        reg1_results = reg1.predict(X_data)
        reg2_results = reg2.predict(X_data)
        
        df = pd.DataFrame(columns = ['target', 'oversampled',
                                     'clf', 'reg1', 'reg2',
                                     'reg0_is_post', 'reg1_is_post', 'reg2_is_post',
                                     'mae', 'amae'])
        
        for m0, m1, m2 in product([0, 1],
                                  [0, 1],
                                  [0, 1]):
            
            # class 0 prediction
            inx = clf_results[1] == 0
            y_obs = y_data[inx]
            if m0 == 0: y_hat = 0 * y_data[inx]
            else: y_hat = 0                 * clf_results[0][inx, 0] + \
                          reg1_results[inx] * clf_results[0][inx, 1] + \
                          reg2_results[inx] * clf_results[0][inx, 2]
            
            # class 1 prediction
            inx = clf_results[1] == 1
            y_obs = np.hstack([y_obs, y_data[inx]])
            if m1 == 0: y_aux = reg1_results[inx]
            else: y_aux = 0                 * clf_results[0][inx, 0] + \
                          reg1_results[inx] * clf_results[0][inx, 1] + \
                          reg2_results[inx] * clf_results[0][inx, 2]
            
            y_hat = np.hstack([y_hat, y_aux])
            
            # class 2 prediction
            inx = clf_results[1] == 2
            y_obs = np.hstack([y_obs, y_data[inx]])
            if m2 == 0: y_aux = reg2_results[inx]
            else: y_aux = 0                 * clf_results[0][inx, 0] + \
                          reg1_results[inx] * clf_results[0][inx, 1] + \
                          reg2_results[inx] * clf_results[0][inx, 2]
            
            y_hat = np.hstack([y_hat, y_aux])

            final_mae = MAE(y_hat, y_obs)
            final_amae = AMAE(y_obs, y_hat, points = 1000)

            row = [
                target,
                oversampled == 1,
                clf_name,
                reg1_name,
                reg2_name,
                m0 == 1,
                m1 == 1,
                m2 == 1,
                final_mae,
                final_amae
            ]

            df.loc[len(df)] = row

    df.to_csv(f'results_final.csv', index = False)