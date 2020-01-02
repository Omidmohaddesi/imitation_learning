from dataset import CrispDataset
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import tkinter
import matplotlib
import gym
from gym_crisp.envs import CrispEnv
from sklearn.metrics import mean_absolute_error as mae

matplotlib.use('TkAgg')

def fun(w, x, y):
    f = w[0] * x[:, 0] + w[1] * x[:, 1] + w[2] * x[:, 2] + w[3] * x[:, 3]
    f[f < 0] = 0
    f.astype(int)
    return f - y


def calculate_order(x, w1, w2, w3, w4, method='multiple'):
    # for calculating multiple observation:
    if method == 'multiple':
        f = (w1 * x[:, 0] + w2 * x[:, 1] + w3 * x[:, 2] + w4 * x[:, 3]).astype(int)
        f[f < 0] = 0
        return f
    # for calculating single observation:
    elif method == 'single':
        return max(0, int(w1 * x[0] + w2 * x[1] + w3 * x[2] + w4 * x[3]))


if __name__ == '__main__':

    expert_data_path = "datasets/player_state_actions/"

    dataset = CrispDataset(expert_data_path)
    order = dataset.order
    inventory = dataset.inventory
    demand = dataset.demand
    backlog = dataset.backlog
    shipment = dataset.shipment

    data = pd.DataFrame(columns=['order', 'inventory', 'shipment', 'demand', 'backlog'])

    # players_to_ignore = [10, 19, 11, 14]
    players_to_ignore = [2, 4, 1, 5, 10, 15]

    for i in range(46, 68):
        if i not in [x + 46 for x in players_to_ignore]:
            data = data.append(
                pd.concat([order.iloc[i, 0:20], inventory.iloc[i, 0:20], shipment.iloc[i, 0:20],
                          demand.iloc[i, 0:20], backlog.iloc[i, 0:20]],
                          axis=1,
                          keys=['order', 'inventory', 'shipment', 'demand', 'backlog']), ignore_index=True)

    models = np.empty(shape=(0, 4), dtype=float)

    for i in range(0, 16*20, 20):

        w0 = np.ones(4)
        y_train = data.iloc[i:i+20, 0].to_numpy(dtype=int)
        x_train = data.iloc[i:i+20, 1:5].to_numpy(dtype=int)

        # res_lsq = least_squares(fun, w0, args=(x_train, y_train))
        res_robust = least_squares(fun, w0, loss='soft_l1', f_scale=0.1, args=(x_train, y_train))

        models = np.append(models, [res_robust.x], axis=0)
        # y_test = data.iloc[i:i+20, 0].to_numpy(dtype=int)
        # x_test = data.iloc[i:i+20, 1:5].to_numpy(dtype=int)

        # y_lsq = calculate_order(x_test, *res_lsq.x)
        # y_robust = calculate_order(x_test, *res_robust.x)

    np.savez('regression_models_2.npz', *models)

    # y_train = pd.DataFrame(np.array_split(y_train, 68))
    # y_lsq = pd.DataFrame(np.array_split(y_lsq, 68))
    # y_robust = pd.DataFrame(np.array_split(y_robust, 68))

    # mean_lsq = y_lsq.mean()
    # mean_robust = y_robust.mean()
    # mean_y = y_train.mean()

    # fig, ax = plt.subplots()
    # boxplot = y_train.boxplot()
    # boxplot.plot(np.array(range(1, 21)), mean_lsq, label='lsq')
    # boxplot.plot(np.array(range(1, 21)), mean_robust, label='robust')
    # plt.xlabel('$t$')
    # plt.ylabel('$order$')
    # plt.legend()

    # fig2, ax2 = plt.subplots()
    # ax2 = plt.plot(mean_robust, mean_y, 'o', label='robust')
    # ax2 = plt.plot(mean_robust, mean_robust)
    # plt.show()

    # env = gym.make('Crisp-v0')
    # env.seed(123)
    # obs = env.reset()
    # reward_sum = 0
    # for _ in range(21):
    #     action = calculate_order(obs, *res_robust.x, method='single')
    #     obs, reward, done, info = env.step(action)
    #     print(action)
    #     reward_sum += reward
    #     # if done:
    #     #     obs = env.reset()
    #     #     reward_sum = 0
    # print(f'Total reward is ', reward_sum)

    # print('lsq: ', mae(y_train, y_lsq))
    # print('res_robust: ', mae(y_train, y_robust))

    # plt.plot(np.array(range(1, 21)), y_train, label='data')
    # plt.plot(np.array(range(1, 21)), y_lsq, label=f'lsq ({mae(y_train, y_lsq)})')
    # plt.plot(np.array(range(1, 21)), y_robust, label=f'robust ({ mae(y_train, y_robust)})')
    # plt.xlabel('$t$')
    # plt.ylabel('$order$')
    # plt.legend()
    # plt.show()


