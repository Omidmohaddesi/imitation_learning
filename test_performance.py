import gym
from stable_baselines import GAIL
from gym_crisp.envs import CrispEnv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    model = GAIL.load("gail_crisp_33")

    env = gym.make('Crisp-v0')
    obs = env.reset()

    # info = {'time': 1}

    prob = []

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
