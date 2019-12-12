import os
import pandas as pd

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

# def fun(x, y, t):



for i in range(0, 68):
    data = pd.concat([order.iloc[i, 0:20], inventory.iloc[i, 0:20], demand.iloc[i, 0:20],
                     backlog.iloc[i, 0:20], shipment.iloc[i, 0:20]],
                     axis=1,
                     keys=['order', 'inventory', 'demand', 'backlog', 'shipment'])

