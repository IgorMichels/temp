import numpy as np
import matplotlib.pyplot as plt

def MAE(y_pred, y_obs):
    return np.mean(np.abs(y_pred - y_obs))

def AMAE(y_obs, y_pred, points = 1000, show = True):
    limits = np.linspace(0, 100, points)
    dx = limits[1] - limits[0]
    maes = list()
    for x in limits:
        inx = y_obs >= x
        maes.append(MAE(y_pred[inx], y_obs[inx]))

    if show:
        plt.plot(limits, maes)
        plt.xlabel('Threshold')
        plt.ylabel('MAE')
        plt.show()

    return np.sum(maes) * dx

if __name__ == '__main__':
    y_pred = np.random.random(191673)
    y_obs = np.random.random(191673)

    indx = np.argsort(y_pred)
    y_pred = y_pred[indx]
    y_obs = y_obs[indx]

    y_pred = y_pred * 100
    y_obs = y_obs / np.max(y_obs) * 100

    print(AMAE(y_obs, y_pred, show = True))