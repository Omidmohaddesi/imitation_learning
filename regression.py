import os
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

expert_data_path = "C:/Users/mohaddesi.s/Documents/PycharmProjects/MyFirstProgram/crisp_game_server" \
                   "/gamette_experiments/study_1/player_state_actions/"

order = pd.read_csv(os.path.join(expert_data_path, 'order_data.csv'), index_col=0)
cost = pd.read_csv(os.path.join(expert_data_path, 'cost_data.csv'), index_col=0)
inventory = pd.read_csv(os.path.join(expert_data_path, 'inventory_data.csv'), index_col=0)
demand = pd.read_csv(os.path.join(expert_data_path, 'demand_data.csv'), index_col=0)
backlog = pd.read_csv(os.path.join(expert_data_path, 'backlog_data.csv'), index_col=0)
shipment = pd.read_csv(os.path.join(expert_data_path, 'shipments_data.csv'), index_col=0)

order = order.reset_index(drop=True)
cost = cost.reset_index(drop=True)
inventory = inventory.reset_index(drop=True)
demand = demand.reset_index(drop=True)
backlog = backlog.reset_index(drop=True)
shipment = shipment.reset_index(drop=True)


def fun(w, x, y):
    return w[0] * x[:, 0] + w[1] * x[:, 1] + w[2] * x[:, 2] + w[3] * x[:, 3] - y


def calculate_order(x, w1, w2, w3, w4):
    return w1 * x[:, 0] + w2 * x[:, 1] + w3 * x[:, 2] + w4 * x[:, 3]


np.linspace(0, 30)
data = pd.DataFrame(columns=['order', 'inventory', 'demand', 'backlog', 'shipment'])

for i in range(0, 68):
    data = data.append(
        pd.concat([order.iloc[i, 0:20], inventory.iloc[i, 0:20], demand.iloc[i, 0:20],
                   backlog.iloc[i, 0:20], shipment.iloc[i, 0:20]],
                  axis=1,
                  keys=['order', 'inventory', 'demand', 'backlog', 'shipment']), ignore_index=True)

w0 = np.ones(4)
y_train = data.iloc[:, 0].to_numpy(dtype=int)
x_train = data.iloc[:, 1:5].to_numpy(dtype=int)

res_lsq = least_squares(fun, w0, args=(x_train, y_train))
res_robust = least_squares(fun, w0, loss='soft_l1', f_scale=0.1, args=(x_train, y_train))

y_test = data.iloc[:, 0].to_numpy(dtype=int)
x_test = data.iloc[:, 1:5].to_numpy(dtype=int)

y_lsq = calculate_order(x_test, *res_lsq.x)
y_robust = calculate_order(x_test, *res_robust.x)

y_train = pd.DataFrame(np.array_split(y_train, 68))
y_lsq = pd.DataFrame(np.array_split(y_lsq, 68))
y_robust = pd.DataFrame(np.array_split(y_robust, 68))

mean_lsq = y_lsq.mean()
mean_robust = y_robust.mean()
mean_y = y_train.mean()

fig, ax = plt.subplots()
boxplot = y_train.boxplot()
boxplot.plot(np.array(range(1, 21)), mean_lsq, label='lsq')
boxplot.plot(np.array(range(1, 21)), mean_robust, label='robust')
plt.xlabel('$t$')
plt.ylabel('$order$')
plt.legend()

fig2, ax2 = plt.subplots()
ax2 = plt.plot(mean_robust, mean_y, 'o', label='robust')
plt.show()

# plt.plot(np.array(range(1, 21)), y_train, 'o', label='data')
# plt.plot(np.array(range(1, 21)), y_lsq, label='lsq')
# plt.plot(np.array(range(1, 21)), y_robust, label='robust')
# plt.xlabel('$t$')
# plt.ylabel('$order$')
# plt.legend()
# plt.show()
#
# print("")
