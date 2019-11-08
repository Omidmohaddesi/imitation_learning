# import gym
#
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
#
# env = gym.make('CartPole-v1')
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
#
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)
#
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

from numpy import load
import numpy as np
import os
import csv
import gym

from stable_baselines import GAIL, SAC
from stable_baselines.gail import ExpertDataset, generate_expert_traj

path = "C:/Users/mohaddesi.s/Downloads/expert_cartpole.npz"

env = gym.make('CartPole-v0')
data = load('expert_cartpole.npz')
# data = load('expert_pendulum.npz')
lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])

expert_data_path = "C:/Users/mohaddesi.s/Documents/PycharmProjects/MyFirstProgram/crisp_game_server" \
                   "/gamette_experiments/study_1/player_state_actions/"

data = {'actions': np.empty((0, 1), int),
        'episode_returns': np.empty((0, 0), int),
        'rewards': np.empty((0, 0), int),
        'obs': np.empty((0, 4), int),
        'episode_starts': np.empty((0, 0), bool)
        }

with open(os.path.join(expert_data_path, 'order_data.csv')) as order_file, \
        open(os.path.join(expert_data_path, 'cost_data.csv')) as cost_file, \
        open(os.path.join(expert_data_path, 'inventory_data.csv')) as inventory_file, \
        open(os.path.join(expert_data_path, 'demand_data.csv')) as demand_file, \
        open(os.path.join(expert_data_path, 'backlog_data.csv')) as backlog_file, \
        open(os.path.join(expert_data_path, 'shipments_data.csv')) as shipments_file:

    order_data = csv.reader(order_file, delimiter=',')
    cost_data = csv.reader(cost_file, delimiter=',')
    inventory_data = csv.reader(inventory_file, delimiter=',')
    demand_data = csv.reader(demand_file, delimiter=',')
    backlog_data = csv.reader(backlog_file, delimiter=',')
    shipments_data = csv.reader(shipments_file, delimiter=',')

    line_count = 0
    for row in order_data:
        elem_count = 1
        if line_count == 0:
            line_count += 1
            pass
        else:
            line_count += 1
            for elem in row[1:-2]:
                data['actions'] = np.append(data['actions'], [[int(elem)]], axis=0)
                if elem_count == 1:
                    data['episode_starts'] = np.append(data['episode_starts'], True)
                    elem_count += 1
                else:
                    data['episode_starts'] = np.append(data['episode_starts'], False)
                    elem_count += 1

    line_count = 0
    for row in cost_data:
        if line_count == 0:
            line_count += 1
            pass
        else:
            line_count += 1
            data['episode_returns'] = np.append(data['episode_returns'], [- int(row[-2])])
            for elem in row[1:-1]:
                data['rewards'] = np.append(data['rewards'], [[- int(elem)]])

    line_count = 0
    for row1, row2, row3, row4 in zip(inventory_data, shipments_data, demand_data, backlog_data):
        if line_count == 0:
            line_count += 1
            pass
        else:
            line_count += 1
            for elem1, elem2, elem3, elem4 in zip(row1[1:-2], row2[1:-2], row3[1:-2], row4[1:-2]):
                data['obs'] = np.append(data['obs'], [[int(elem1), int(elem2), int(elem3), int(elem4)]], axis=0)

# np.savez('expert_data.npz', data['actions'], data['episode_returns'],
#          data['obs'], data['rewards'], data['episode_starts'])
np.savez('expert_data.npz', **data)

# Generate expert trajectories (train expert)
# model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
# generate_expert_traj(model, 'expert_pendulum', n_timesteps=100, n_episodes=10)

# Load the expert dataset
dataset = ExpertDataset(expert_path='expert_data.npz', verbose=1)

model = GAIL("MlpPolicy", 'Inventory-v0', dataset, verbose=1)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=1000)
model.save("gail_crisp")

del model # remove to demonstrate saving and loading

model = GAIL.load("gail_crisp")

env = gym.make('crisp-v0')
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
