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

    df = pd.DataFrame(columns=['num_traj', 'performance', 'data_type'])

    players = [8, 15, 0, 17, 11, 12, 7, 2, 10, 16, 5, 6, 1, 9, 14, 13, 4, 3]

    for i in num_traj:
        df = df.append({'num_traj': i,
                        'performance': data['episode_returns'][players[0:i]].mean(),
                        'data_type': 'Expert'}, ignore_index=True)
        # df = df.append({'num_traj': i,
        #                 'performance': data['episode_returns'][0:i].mean(),
        #                 'data_type': 'Expert'}, ignore_index=True)

    for i in num_traj:
        env = gym.make('Crisp-v0')
        env.seed(123)
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

    regressions = np.load('regression_models.npz')

    for i in num_traj:
        env = gym.make('Crisp-v0')
        env.seed(123)
        for j in players[0:i]:
            weights = regressions.get(f'arr_{j}')
            obs = env.reset()
            reward_sum = 0
            for _ in range(21):
                action = calculate_order(obs, *weights, method='single')
                obs, reward, done, info = env.step(action)
                reward_sum += reward
            df = df.append({'num_traj': i,
                            'performance': reward_sum,
                            'data_type': 'Regression'}, ignore_index=True)

    gail_models = {i: GAIL.load(f'./models/with_sorted_performance/gail_crisp_{i}') for i in num_traj}
    bc_models = {i: GAIL.load(f'./models/with_sorted_performance/BC_crisp_{i}') for i in num_traj}

    for item in gail_models.items():
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
        print(f'GAIL model {item[0]} done!')

    for item in bc_models.items():
        model = item[1]
        env = gym.make('Crisp-v0')
        obs = env.reset()
        reward_sum = 0
        for _ in range(1, 1000):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                obs = env.reset()
                df = df.append({'num_traj': item[0],
                                'performance': reward_sum,
                                'data_type': 'Behavioral Cloning'}, ignore_index=True)
                reward_sum = 0
        print(f'Behavioral Cloning model {item[0]} done!')

    df2 = df.copy()
    for i in num_traj:
        min_n = df2[(df2['num_traj'] == i) & (df2['data_type'] == 'Random')]['performance'].values[0]
        max_n = df2[(df2['num_traj'] == i) & (df2['data_type'] == 'Expert')]['performance'].values[0]
        df.performance[df['num_traj'] == i] = df.performance[df['num_traj'] == i].apply(
            lambda x: (x - min_n) / (max_n - min_n))

    fig = plt.figure()
    plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.13, wspace=0.2, hspace=0.4)
    fig.set_size_inches(10, 6)
    sns.set()
    sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 2.5, "lines.markeredgewidth": 1})
    ax = sns.lineplot(x='num_traj', y='performance', hue='data_type', data=df, style='data_type',
                      style_order=['GAIL', 'Expert', 'Random', 'Behavioral Cloning', 'Regression'])
    ax.get_legend().set_title('')
    # new_labels = ['', 'Expert', 'Random', 'GAIL', 'Behavioral Cloning', 'Regression']
    new_labels = ['', 'Expert', 'Random', 'Regression', 'GAIL', 'Behavioral Cloning']
    for t, l in zip(ax.get_legend().texts, new_labels):
        t.set_text(l)
    ax.set(xticks=num_traj, ylabel='Performance', xlabel='Number of trajectories in dataset')
    sns.despine(offset=5, trim=True)
    fig.savefig('expert_vs_gail2.png', format='png', dpi=300)

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
