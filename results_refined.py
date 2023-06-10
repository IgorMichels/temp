import numpy as np
import pandas as pd
from glob import glob

import matplotlib.pyplot as plt

if __name__ == '__main__':
    files = glob('results_refined/*')
    df = pd.DataFrame()
    for file in files: df = pd.concat([df, pd.read_csv(file)], ignore_index = True)
    
    maes = list()
    amaes = list()
    for t in range(1, 5):
        maes.append(df[df['target'] == t].sort_values('mae', ignore_index = True).loc[0, 'mae'])
        amaes.append(df[df['target'] == t].sort_values('mae', ignore_index = True).loc[0, 'amae'])

    print(f'MAE: {np.mean(maes)}')
    print(f'AMAE: {np.mean(amaes)}')

    fig = plt.figure(layout = 'constrained')
    ax = fig.subplot_mosaic('''AB
                               CD''')
    
    for target in range(1, 5):
        aux = df[df['target'] == target]
        aux = aux.sort_values('mae').reset_index(drop = True)
        oversampled = aux.loc[0, 'oversampled']
        clf = aux.loc[0, 'clf']
        reg1 = aux.loc[0, 'reg1']
        reg2 = aux.loc[0, 'reg2']
        reg0_post = aux.loc[0, 'reg0_is_post']
        reg1_post = aux.loc[0, 'reg1_is_post']
        reg2_post = aux.loc[0, 'reg2_is_post']
        aux = aux[((aux['oversampled'] == oversampled) & \
                   (aux['clf'] == clf) & \
                   (aux['reg1'] == reg1) & \
                   (aux['reg2'] == reg2) & \
                   (aux['reg0_is_post'] == reg0_post) & \
                   (aux['reg1_is_post'] == reg1_post) & \
                   (aux['reg2_is_post'] == reg2_post))]
        
        aux = aux.sort_values('slice_point', ignore_index = True)
        graph = chr(64 + target)
        ax[graph].plot(aux['slice_point'], aux['mae'])
        ax[graph].set_xlabel('slice_point')
        ax[graph].set_ylabel('MAE')
        ax[graph].set_title(f'Target{target}')
    
    plt.savefig('mae_2.png')
    plt.clf()

    fig = plt.figure(layout = 'constrained')
    ax = fig.subplot_mosaic('''AB
                               CD''')
    
    for target in range(1, 5):
        aux = df[df['target'] == target]
        aux = aux.sort_values('amae').reset_index(drop = True)
        oversampled = aux.loc[0, 'oversampled']
        clf = aux.loc[0, 'clf']
        reg1 = aux.loc[0, 'reg1']
        reg2 = aux.loc[0, 'reg2']
        reg0_post = aux.loc[0, 'reg0_is_post']
        reg1_post = aux.loc[0, 'reg1_is_post']
        reg2_post = aux.loc[0, 'reg2_is_post']
        aux = aux[((aux['oversampled'] == oversampled) & \
                   (aux['clf'] == clf) & \
                   (aux['reg1'] == reg1) & \
                   (aux['reg2'] == reg2) & \
                   (aux['reg0_is_post'] == reg0_post) & \
                   (aux['reg1_is_post'] == reg1_post) & \
                   (aux['reg2_is_post'] == reg2_post))]
        
        aux = aux.sort_values('slice_point', ignore_index = True)
        graph = chr(64 + target)
        ax[graph].plot(aux['slice_point'], aux['amae'])
        ax[graph].set_xlabel('slice_point')
        ax[graph].set_ylabel('AMAE')
        ax[graph].set_title(f'Target{target}')
    
    plt.savefig('amae_2.png')
    plt.clf()