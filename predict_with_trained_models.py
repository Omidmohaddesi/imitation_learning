import gym
from stable_baselines import GAIL
from gym_crisp.envs import CrispEnv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from regression import calculate_order


def scale_performance(y, min_n, max_n):
    return (y - min_n) / (max_n - min_n)


if __name__ == '__main__':

    data = np.load('expert_data.npz')

    num_traj = [1, 6, 12, 18]

    df = pd.DataFrame(columns=['week', 'order', 'data_type'])

    players = [8, 15, 0, 17, 11, 12, 7, 2, 10, 16, 5, 6, 1, 9, 14, 13, 4, 3]

    i = 1
    for j in range(len(data['actions'])):
        df = df.append({'week': i,
                        # 'order': data['actions'][j][0],
                        'order': data['actions'][players[0] * 20 + i - 1][0],
                        'data_type': 'Expert'
                        }, ignore_index=True)
        if i == 20:
            i = 1
            print('Expert: ', data['episode_returns'][players[0]])
            break
        else:
            i += 1

    # gail_models = {i: GAIL.load(f'./models/with_sorted_performance/1/gail_crisp_{i}') for i in num_traj}
    model = GAIL.load('./models/with_sorted_performance/1/gail_crisp_1')

    env = gym.make('Crisp-v0')
    obs = env.reset()
    prob = []
    reward_list = []
    i = 1
    reward_sum = 0
    for j in range(1000):
        action, _states = model.predict(obs)
        df = df.append({'week': i,
                        'order': action,
                        'data_type': 'GAIL'}, ignore_index=True)
        # prob.append(model.action_probability(obs))
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        i += 1
        if done:
            obs = env.reset()
            reward_list.append(reward_sum)
            reward_sum = 0
            i = 1
            # print('GAIL: ', reward_sum)
    print('GAIL: ', sum(reward_list)/len(reward_list))

    bc_model = GAIL.load('./models/with_sorted_performance/1/BC_crisp_1')

    env = gym.make('Crisp-v0')
    obs = env.reset()
    prob = []
    reward_list = []
    i = 1
    reward_sum = 0
    for j in range(1000):
        action, _states = bc_model.predict(obs)
        df = df.append({'week': i,
                        'order': action,
                        'data_type': 'Behavioral Cloning'}, ignore_index=True)
        # prob.append(model.action_probability(obs))
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        i += 1
        if done:
            obs = env.reset()
            reward_list.append(reward_sum)
            reward_sum = 0
            i = 1
            # print('GAIL: ', reward_sum)
    print('Behavioral Cloning: ', sum(reward_list) / len(reward_list))

    regressions = np.load('regression_models.npz')

    env = gym.make('Crisp-v0')
    env.seed(123)
    for j in players[0:]:
        if j == 8:
            weights = regressions.get(f'arr_{j}')
            obs = env.reset()
            reward_sum = 0
            for i in range(21):
                action = calculate_order(obs, *weights, method='single')
                df = df.append({'week': i,
                                'order': action,
                                'data_type': 'Regression'}, ignore_index=True)
                obs, reward, done, info = env.step(action)
                reward_sum += reward
            print('Regression: ', reward_sum)

    df.week = df.week.astype(int)
    df.order = df.order.astype(int)

    fig = plt.figure()
    plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.13, wspace=0.2, hspace=0.4)
    fig.set_size_inches(10, 6)
    sns.set()
    sns.set_context("paper")
    ax = sns.lineplot(x='week', y='order', hue='data_type', data=df, style='data_type')
    ax.set(xticks=range(1, 21), ylabel='Order Amount', xlabel='Week No.')
    plt.show()
