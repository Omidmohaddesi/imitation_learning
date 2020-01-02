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

    rows_to_ignore = [10, 12, 14, 20]
                      # 9,
                      # 4, 3, 13, 11, 23, 8,
                      # 19, 15, 21, 18, 17, 16,
                      # 6, 22, 1, 7, 0, 5, 2]

    '''
        # for condition 2 OUL w/o suggestion
        # [2, 4, 1, 5, 10, 15]  
                      # 0,
                      # 19, 14, 17, 21,
                      # 3, 9, 8, 20, 7,
                      # 13, 12, 6, 11, 18, 16]

        for condition 1 human like
        # 11, 20, 12, 15]   # players to remove
                      # 9,
                      # 19, 1, 22, 14, 16,
                      # 8, 3, 13, 21, 6, 7,
                      # 2, 10, 18, 17, 5, 4]

                      # 2, 3, 4, 5, 6,  # before
                      # 7, 8, 9, 10, 13, 14,
                      # 16, 17, 18, 19, 21, 22]
    '''
    expert_data_path = "datasets/player_state_actions/"

    data = {'actions': np.empty((0, 1), int),
            'episode_returns': np.empty((0, 0), int),
            'rewards': np.empty((0, 0), int),
            'obs': np.empty((0, 7), int),
            'episode_starts': np.empty((0, 0), bool)
            }

    with open(os.path.join(expert_data_path, 'order_data.csv')) as order_file, \
            open(os.path.join(expert_data_path, 'cost_data2.csv')) as cost_file, \
            open(os.path.join(expert_data_path, 'inventory_data.csv')) as inventory_file, \
            open(os.path.join(expert_data_path, 'demand_data.csv')) as demand_file, \
            open(os.path.join(expert_data_path, 'backlog_data.csv')) as backlog_file, \
            open(os.path.join(expert_data_path, 'shipments_data.csv')) as shipments_file, \
            open(os.path.join(expert_data_path, '3_up_to_level.csv')) as upToLevel_file, \
            open(os.path.join(expert_data_path, '3_on_order.csv')) as onOrder_file, \
            open(os.path.join(expert_data_path, '3_suggested.csv')) as suggested_file:

        order_data = csv.reader(order_file, delimiter=',')
        cost_data = csv.reader(cost_file, delimiter=',')
        inventory_data = csv.reader(inventory_file, delimiter=',')
        demand_data = csv.reader(demand_file, delimiter=',')
        backlog_data = csv.reader(backlog_file, delimiter=',')
        shipments_data = csv.reader(shipments_file, delimiter=',')
        upToLevel_data = csv.reader(upToLevel_file, delimiter=',')
        onOrder_data = csv.reader(onOrder_file, delimiter=',')
        suggested_data = csv.reader(suggested_file, delimiter=',')

        line_count = 0
        for row in order_data:
            elem_count = 1
            # if line_count == 0:
            if line_count < 23 or line_count > 46 or line_count in [x + 23 for x in rows_to_ignore]:  # Only considering beerGame condition
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
            if line_count < 23 or line_count > 46 or line_count in [x + 23 for x in rows_to_ignore]:  # Only considering beerGame condition
                line_count += 1
                pass
            else:
                line_count += 1
                data['episode_returns'] = np.append(data['episode_returns'], [-sum(list(map(int, row[1:-1])))])
                for elem in row[1:-1]:
                    data['rewards'] = np.append(data['rewards'], [[- int(elem)]])

        line_count = 0
        line_count_2 = 0
        for row1, row2, row3, row4 in zip(inventory_data, shipments_data, demand_data, backlog_data):
            # if line_count == 0:
            if line_count < 23 or line_count > 46 or line_count in [x + 23 for x in rows_to_ignore]:  # Only considering beerGame condition
                line_count += 1
                pass
            else:
                line_count += 1
                for row5, row6, row7 in zip(upToLevel_data, onOrder_data, suggested_data):
                    if line_count_2 == 0 or line_count_2 in [y + 1 for y in rows_to_ignore]:
                        line_count_2 += 1
                        pass
                    else:
                        line_count_2 += 1
                        for elem1, elem2, elem3, elem4, elem5, elem6, elem7 in zip(row1[1:-2], row2[1:-2], row3[1:-2],
                                                                                   row4[1:-2], row5[1:], row6[1:],
                                                                                   row7[1:]):
                            data['obs'] = np.append(data['obs'], [[int(elem1), int(elem2), int(elem3), int(elem4),
                                                                   int(elem5), int(elem6), int(elem7)]], axis=0)

    np.savez('expert_data_3.npz', **data)

    # Create and wrap the environment
    env = gym.make('Crisp-v0')

    # Generate expert trajectories (train expert)
    # model = A2C('MlpPolicy', 'Crisp-v0', verbose=1)
    # generate_expert_traj(model, 'Crisp-v0', n_timesteps=20, n_episodes=100)

    # tf.debugging.set_log_device_placement(True)

    # Load the expert dataset
    dataset = ExpertDataset(expert_path='expert_data_3.npz', verbose=1)

    model = GAIL("MlpPolicy", env, dataset, verbose=2,
                 tensorboard_log='./tmp/gail/5/3',
                 full_tensorboard_log=True,
                 timesteps_per_batch=1000,
                 )

    # Note: in practice, you need to train for 1M steps to have a working policy
    model.pretrain(dataset, n_epochs=5000, learning_rate=1e-5, adam_epsilon=1e-08, val_interval=None)
    # model.learn(total_timesteps=100000, callback=callback)
    params = model.get_parameters()
    model.save("./models/with_sorted_performance/3/BC_crisp_20")

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
