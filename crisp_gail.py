import numpy as np
import os
import csv
import gym
import gym_crisp
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


from stable_baselines import SAC, PPO2, A2C, DQN
from stable_baselines.bench import Monitor
from stable_baselines.gail import GAIL
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from gym_crisp.envs import CrispEnv


def callback(locals_, globals_):
    self_ = locals_['self']
    seg_ = locals_['seg_gen'].__next__()
    true_rewrds_ = seg_['true_rewards']
    mask_ = seg_['dones']
    value = sum(true_rewrds_) / sum(mask_)
    # value = self_.episode_reward
    summary = tf.Summary(value=[tf.Summary.Value(tag='Average_Episodes_Reward', simple_value=value)])
    locals_['writer'].add_summary(summary, self_.num_timesteps)
    return True


if __name__ == '__main__':

    # path = "C:/Users/mohaddesi.s/Downloads/expert_cartpole.npz"

    # env = gym.make('Crisp-v0')
    # data = load('expert_cartpole.npz')
    # data = load('expert_pendulum.npz')
    # lst = data.files
    # for item in lst:
    #     print(item)
    #     print(data[item])

    expert_data_path = "datasets/player_state_actions/"

    data = {'actions': np.empty((0, 1), int),
            'episode_returns': np.empty((0, 0), int),
            'rewards': np.empty((0, 0), int),
            'obs': np.empty((0, 4), int),
            'episode_starts': np.empty((0, 0), bool)
            }

    with open(os.path.join(expert_data_path, 'order_data.csv')) as order_file, \
            open(os.path.join(expert_data_path, 'cost_data2.csv')) as cost_file, \
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
            # if line_count == 0:
            if line_count == 0 or line_count > 22 or line_count in [11, 20, 12, 15]:  # Only considering beerGame condition
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
            # if line_count == 0:
            if line_count == 0 or line_count > 22 or line_count in [11, 20, 12, 15]:  # Only considering beerGame condition
                line_count += 1
                pass
            else:
                line_count += 1
                data['episode_returns'] = np.append(data['episode_returns'], [-sum(list(map(int, row[1:-1])))])
                for elem in row[1:-1]:
                    data['rewards'] = np.append(data['rewards'], [[- int(elem)]])

        line_count = 0
        for row1, row2, row3, row4 in zip(inventory_data, shipments_data, demand_data, backlog_data):
            # if line_count == 0:
            if line_count == 0 or line_count > 22 or line_count in [11, 20, 12, 15]:  # Only considering beerGame condition
                line_count += 1
                pass
            else:
                line_count += 1
                for elem1, elem2, elem3, elem4 in zip(row1[1:-2], row2[1:-2], row3[1:-2], row4[1:-2]):
                    data['obs'] = np.append(data['obs'], [[int(elem1), int(elem2), int(elem3), int(elem4)]], axis=0)


    # np.savez('expert_data.npz', data['actions'], data['episode_returns'],
    #          data['obs'], data['rewards'], data['episode_starts'])
    np.savez('expert_data.npz', **data)

    # Create log dir
    log_dir = './tmp/gail/3'
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make('Crisp-v0')
    # Logs will be saved in log_dir/monitor.csv
    # env = Monitor(env, log_dir, allow_early_resets=True)

    # Generate expert trajectories (train expert)
    # model = A2C('MlpPolicy', 'Crisp-v0', verbose=1)
    # generate_expert_traj(model, 'Crisp-v0', n_timesteps=20, n_episodes=100)

    # tf.debugging.set_log_device_placement(True)

    # Load the expert dataset
    dataset = ExpertDataset(expert_path='expert_data.npz', verbose=1)

    model = GAIL("MlpPolicy", env, dataset, verbose=2,
                 tensorboard_log='./tmp/gail/2',
                 full_tensorboard_log=True,
                 timesteps_per_batch=1000,
                 )

    # Note: in practice, you need to train for 1M steps to have a working policy
    # model.pretrain(dataset, n_epochs=2000, learning_rate=1e-5, adam_epsilon=1e-08, val_interval=None)
    model.learn(total_timesteps=60000, callback=callback)
    params = model.get_parameters()
    model.save("gail_crisp_33")

    # del model # remove to demonstrate saving and loading

    # model = GAIL.load("gail_crisp")

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

    # fig = plt.figure()
    # fig.set_size_inches(10, 6)
    # sns.set()
    # sns.set_context("paper")
    # ax = sns.lineplot(np.arange(0, 500), prob[0])
    # fig.savefig('dist2.png', format='png', dpi=300)

    # env = DummyVecEnv([lambda: CrispEnv()])
    # model = PPO2(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=40000)
    # obs = env.reset()
    # for i in range(21):
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     env.render()
