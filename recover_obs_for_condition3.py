import gym
from gym_crisp.envs import CrispEnv
import pandas as pd
from dataset import CrispDataset


expert_data_path = "datasets/player_state_actions/"

dataset = CrispDataset(expert_data_path)
order = dataset.order.iloc[22:46, 0:20].reset_index(drop=True)

up_to_level = pd.DataFrame().reindex_like(order)
on_order = pd.DataFrame().reindex_like(order)
suggested = pd.DataFrame().reindex_like(order)

env = gym.make('Crisp-v0')
obs = env.reset()

for i in range(len(order)):
    for j in range(1, 21):
        up_to_level.at[i, f'{j}'] = obs[4]
        on_order.at[i, f'{j}'] = obs[5]
        suggested.at[i, f'{j}'] = obs[6]
        action = order.at[i, f'{j}']
        obs, reward, done, info = env.step(action)
        if done:
            print(f'Player {i} is done! \n')
            obs = env.reset()

up_to_level = up_to_level.astype(int)
on_order = on_order.astype(int)
suggested = suggested.astype(int)

up_to_level.to_csv('datasets/player_state_actions/3_up_to_level.csv')
on_order.to_csv('datasets/player_state_actions/3_on_order.csv')
suggested.to_csv('datasets/player_state_actions/3_suggested.csv')
