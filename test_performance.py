import gym
from stable_baselines import GAIL
from gym_crisp.envs import CrispEnv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scale_performance(y, min_n, max_n):
    return (y - min_n) / (max_n - min_n)


if __name__ == '__main__':

    data = np.load('expert_data.npz')

    num_traj = [1, 6, 12, 18]

    df = pd.DataFrame(columns=['num_traj', 'performance', 'data_type'])

    for i in num_traj:
        df = df.append({'num_traj': i,
                        'performance': data['episode_returns'][0:i].mean(),
                        'data_type': 'Expert'}, ignore_index=True)

    for i in num_traj:
        env = gym.make('Crisp-v0')
        _ = env.reset()
        reward_sum = 0
        for _ in range(21):
            action = env.action_space.sample()
            _, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                _ = env.reset()
                df = df.append({'num_traj': i,
                                'performance': reward_sum,
                                'data_type': 'Random'}, ignore_index=True)
                reward_sum = 0
        print(f' Random model {i} done!')

    models = {i: GAIL.load(f'./models/gail_crisp_{i}') for i in num_traj}

    for item in models.items():
        model = item[1]
        env = gym.make('Crisp-v0')
        obs = env.reset()
        prob = []
        reward_sum_list = []
        reward_sum = 0
        for _ in range(1, 1000):
            action, _states = model.predict(obs)
            # prob.append(model.action_probability(obs))
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            # env.render()
            if done:
                # print(f'Total reward is {reward_sum} \n')
                obs = env.reset()
                df = df.append({'num_traj': item[0],
                                'performance': reward_sum,
                                'data_type': 'GAIL'}, ignore_index=True)
                reward_sum_list.append(reward_sum)
                reward_sum = 0
        # print('mean is ', np.mean(reward_sum_list))
        print(f'model {item[0]} done!')

    df2 = df.copy()
    for i in num_traj:
        min_n = df2[(df2['num_traj'] == i) & (df2['data_type'] == 'Random')]['performance'].values[0]
        max_n = df2[(df2['num_traj'] == i) & (df2['data_type'] == 'Expert')]['performance'].values[0]
        # for index, row in df[df['num_traj'] == i].iterrows():
        df.performance[df['num_traj'] == i] = df.performance[df['num_traj'] == i].apply(
            lambda x: (x - min_n) / (max_n - min_n))
        # df['performance'] = df['performance'].apply(
        #     lambda x: (x - min_n) / (max_n - min_n) if df['num_traj'].item() == i else x)

    fig = plt.figure()
    fig.set_size_inches(10, 6)
    sns.set()
    sns.set_context("paper")
    ax = sns.lineplot(x='num_traj', y='performance', hue='data_type', data=df)
    ax.set(xticks=num_traj, ylabel='Performance', xlabel='Number of trajectories in dataset')
    fig.savefig('expert_vs_gail.png', format='png', dpi=300)

    '''
    titles = []
    x = []
    observation = data['obs'][40:60, :]
    i = 1
    for obs in observation:
        tmp = model.action_probability(obs)
        prob.extend(list(tmp))
        x.extend(range(500))
        for _ in range(len(tmp)):
            titles.append(f'{i} - ' + str(obs))
        i += 1

    df = pd.DataFrame(dict(x=x, y=prob, g=titles))

    fig = plt.figure()
    fig.set_size_inches(10, 6)
    sns.set()
    sns.set_context("paper")
    sns.lineplot(y='y', x='x', hue='g', data=df)
    fig.savefig('action_prob.png', format='png', dpi=300)

    '''

    # fig = plt.figure()
    # fig.set_size_inches(10, 6)
    # sns.set()
    # sns.set_context("paper")
    # ax = sns.lineplot(np.arange(0, 500), prob[0])
    # fig.savefig('dist3.png', format='png', dpi=300)
    #
