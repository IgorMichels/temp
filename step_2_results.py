import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

if __name__ == '__main__':
    if 'figures' not in os.listdir(): os.mkdir('figures')
    files = glob('results/*')
    df = pd.DataFrame()
    for file in files: df = pd.concat([df, pd.read_csv(file)], ignore_index = True)
    
    maes = list()
    amaes = list()
    for t in range(1, 5):
        maes.append(df[df['target'] == t].sort_values('mae', ignore_index = True).loc[0, 'mae'])
        amaes.append(df[df['target'] == t].sort_values('mae', ignore_index = True).loc[0, 'amae'])

    print(f'MAE: {np.mean(maes):.4f}')
    print(f'AMAE: {np.mean(amaes):.4f}')

    fig = plt.figure(layout = 'constrained')
    ax = fig.subplot_mosaic('''AB
                               CD''')
    
    for target in range(1, 5):
        aux = df[df['target'] == target]
        aux = aux.sort_values('mae').reset_index(drop = True)
        oversampled = aux.loc[0, 'oversampled']
        clf = aux.loc[0, 'clf']
        reg0 = aux.loc[0, 'reg0']
        reg1 = aux.loc[0, 'reg1']
        reg2 = aux.loc[0, 'reg2']
        reg0_post = aux.loc[0, 'reg0_is_post']
        reg1_post = aux.loc[0, 'reg1_is_post']
        reg2_post = aux.loc[0, 'reg2_is_post']
        aux = aux[((aux['oversampled'] == oversampled) & \
                   (aux['clf'] == clf) & \
                   (aux['reg0'] == reg0) & \
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
    
    plt.savefig('figures/mae.png')
    plt.clf()

    fig = plt.figure(layout = 'constrained')
    ax = fig.subplot_mosaic('''AB
                               CD''')
    
    for target in range(1, 5):
        aux = df[df['target'] == target]
        aux = aux.sort_values('amae').reset_index(drop = True)
        oversampled = aux.loc[0, 'oversampled']
        clf = aux.loc[0, 'clf']
        reg0 = aux.loc[0, 'reg0']
        reg1 = aux.loc[0, 'reg1']
        reg2 = aux.loc[0, 'reg2']
        reg0_post = aux.loc[0, 'reg0_is_post']
        reg1_post = aux.loc[0, 'reg1_is_post']
        reg2_post = aux.loc[0, 'reg2_is_post']
        aux = aux[((aux['oversampled'] == oversampled) & \
                   (aux['clf'] == clf) & \
                   (aux['reg0'] == reg0) & \
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
    
    plt.savefig('figures/amae.png')
    plt.clf()
    
    retrain = list()
    vars = np.linspace(-0.45, 0.45, 19)
    for target in range(1, 5):
        aux_target = df[df['target'] == target]
        for metric in ['mae', 'amae']:
            aux = aux_target.sort_values(metric, ignore_index = True).head(4)
            for i in range(4):
                for column in ['clf', 'reg1', 'reg2']:
                    for var in vars:
                        model = [aux.loc[i, 'target'],
                                 aux.loc[i, 'slice_point'] + var,
                                 aux.loc[i, 'oversampled'],
                                 aux.loc[i, column],
                                 column]
                        
                        model = [f'{arg:.2f}' if isinstance(arg, float) else str(arg) for arg in model]
                        if model not in retrain: retrain.append(model)

    with open('retrain.csv', 'w') as f:
        f.write('target,slice_point,oversampled,model_name,model\n')
        for model in retrain: f.write(','.join(model) + '\n')
