import gym
from stable_baselines import GAIL
from gym_crisp.envs import CrispEnv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':

    data = np.load('expert_data.npz')

    model = GAIL.load("gail_crisp_33")

    env = gym.make('Crisp-v0')
    obs = env.reset()

    # info = {'time': 1}

    prob = []
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

    reward_sum = 0
    # while info['time'] < 21:
    for _ in range(1, 1000):
        action, _states = model.predict(obs)
        prob.append(model.action_probability(obs))
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        # env.render()
        if done:
            print(f'Total reward is {reward_sum} \n')
            reward_sum = 0
            obs = env.reset()

    fig = plt.figure()
    fig.set_size_inches(10, 6)
    sns.set()
    sns.set_context("paper")
    ax = sns.lineplot(np.arange(0, 500), prob[0])
    fig.savefig('dist3.png', format='png', dpi=300)
    '''
