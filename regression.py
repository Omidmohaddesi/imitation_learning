from dataset import CrispDataset
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import tkinter
import matplotlib
import gym
# from gym_crisp.envs import CrispEnv
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


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

    data = pd.DataFrame(columns=['order', 'inventory', 'shipment', 'demand', 'backlog', 'player_id'])

    players_to_ignore = [10, 19, 11, 14]

    for i in range(0, 22):
        if i not in players_to_ignore:
            data = data.append(
                pd.concat([order.iloc[i, 0:20].reset_index(drop=True), inventory.iloc[i, 0:20].reset_index(drop=True),
                           shipment.iloc[i, 0:20].reset_index(drop=True), demand.iloc[i, 0:20].reset_index(drop=True),
                           backlog.iloc[i, 0:20].reset_index(drop=True),
                           pd.Series(np.repeat(order.iloc[i, 21], 20))], axis=1,
                          keys=['order', 'inventory', 'shipment', 'demand', 'backlog', 'player_id'])
                , ignore_index=True)

    models = np.empty(shape=(0, 4), dtype=float)

    data.to_csv('datasets/player_data_1.csv')

    n = int(data.shape[0] / 20)
    print('n: ', n)

    error = []
    # error2 = []
    rmse = []
    r2 = []
    adjusted_r2 = []

    for i in range(0, 18*20, 20):

        w0 = np.ones(4)
        y_train = data.iloc[i:i+20, 0].to_numpy(dtype=int)
        x_train = data.iloc[i:i+20, 1:5].to_numpy(dtype=int)

        # res_lsq = least_squares(fun, w0, args=(x_train, y_train))
        res_robust = least_squares(fun, w0, loss='soft_l1', f_scale=0.1, args=(x_train, y_train))

        models = np.append(models, [res_robust.x], axis=0)
        y_test = data.iloc[i:i+20, 0].to_numpy(dtype=int)
        x_test = data.iloc[i:i+20, 1:5].to_numpy(dtype=int)

        # y_lsq = calculate_order(x_test, *res_lsq.x)
        y_robust = calculate_order(x_test, *res_robust.x)

        error.append(mae(y_train, y_robust))
        # error2.append(mae(y_train, y_lsq))
        rmse.append(round(np.sqrt(mse(y_train, y_robust)), 2))
        r = r2_score(y_train, y_robust)
        adj_r = 1 - (1 - r) * (n - 1) / (n - 4 - 1)
        r2.append(round(r, 2))
        adjusted_r2.append(round(adj_r, 2))

    print('robust: ', error)
    # print('lsq:    ', error2)
    print('RMSE:    ', rmse)
    print('R^2:    ', r2)
    print('Adjusted R^2:    ', adjusted_r2)

    print('mean RMSE: ', round(np.mean(rmse), 2))
    print('mean R^2: ', round(np.mean(r2), 2))
    print('median R^2: ', np.median(r2))
    print('mean Adjusted R^2: ', round(np.mean(adjusted_r2), 2))
    print('median Adjusted R^2: ', np.median(adjusted_r2))

    # np.savez('regression_models.npz', *models)

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
