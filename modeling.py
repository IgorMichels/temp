from scripts import load_data, train
import sys
import os

if __name__ == '__main__':
    if 'models' not in os.listdir(): os.mkdir('models')
    X_train, y_train = load_data()
    t = 0
    slice_point = 0
    oversampled = 0
    for arg in sys.argv:
        if '-t' in arg: t = int(arg.split('=')[-1])
        if '-s' in arg: slice_point = float(arg.split('=')[-1])
        if '-o' in arg: oversampled = int(arg.split('=')[-1])

    train(X_train, y_train, t, slice_point, oversampled)