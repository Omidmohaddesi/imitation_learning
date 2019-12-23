from dataset import CrispDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import os
from random import shuffle
import datetime

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps, train_size):
    X, y = [], []
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    x_train, y_train = X[:train_size], y[:train_size]
    x_test, y_test = X[train_size:], y[train_size:]
    return np.array(x_train, dtype=float), np.array(y_train, dtype=float), np.array(x_test, dtype=float), np.array(
        y_test, dtype=float), np.array(X, dtype=float), np.array(y, dtype=float)


class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1, :-1], window[-2, -1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


expert_data_path = "datasets/player_state_actions/"


series = CrispDataset(expert_data_path)
order = series.order
inventory = series.inventory
demand = series.demand
backlog = series.backlog
shipment = series.shipment

n_steps = 4

series = pd.DataFrame(columns=['inventory', 'demand', 'backlog', 'shipment', 'order'])
zero_matrix = pd.DataFrame(np.zeros((10, 5)), columns=['inventory', 'demand', 'backlog', 'shipment', 'order'])

expert_id = list(range(0, 68))
# shuffle(expert_id)
for i in expert_id:
    series = series.append(zero_matrix)
    series = series.append(pd.concat([inventory.iloc[i, 0:20], demand.iloc[i, 0:20],
                                      backlog.iloc[i, 0:20], shipment.iloc[i, 0:20], order.iloc[i, 0:20]],
                                      axis=1,
                                      keys=['inventory', 'demand', 'backlog', 'shipment', 'order']))

series = series.values.astype(int)

split_time = 50 * 30
x_train = series[:split_time, :-1]
y_train = series[:split_time, -1]
x_test = series[split_time:, :-1]
y_test = series[split_time:, -1]

window_size = 4
batch_size = 100
shuffle_buffer = 10

# plt.plot(dataset[:, 4])
# plt.show()
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# n_train_period = 40 * 24
# x_train, y_train, x_test, y_test, X, y = split_sequences(dataset, n_steps, n_train_period)
# n_features = x_train.shape[2]

dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer)
n_features = dataset.element_spec[0].shape[2]

# print(x_train.shape, y_train.shape)

# summarize the data
# for i in range(len(x_train)):
#     print(x_train[i], y_train[i])

callbacks = MyCallBack()

# plt.plot(np.array(range(0, 797)), y)
# plt.show()

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

tf.keras.backend.clear_session()

# define model 1
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(50, activation='relu', input_shape=(n_steps, n_features)),
#     tf.keras.layers.Dense(50, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# define model 2
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(50, activation='relu'),
#     tf.keras.layers.Dense(50, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# define model 3
# model = tf.keras.models.Sequential([
#     tf.keras.layers.SimpleRNN(40, return_sequences=True, input_shape=(None, n_features)),
#     tf.keras.layers.SimpleRNN(40),
#     tf.keras.layers.Dense(1),
#     tf.keras.layers.Lambda(lambda x: x * 100.0)
# ])

# define model 4 (MAE = 39.004642)
# model = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(None, n_features)),
#     tf.keras.layers.LSTM(32),
#     tf.keras.layers.Dense(1),
#     tf.keras.layers.Lambda(lambda x: x * 100.0)
# ])

# define model 5 (MAE = 29.852968)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(500, kernel_size=2, activation=tf.keras.layers.LeakyReLU(),
                           input_shape=(None, n_features)),
    tf.keras.layers.LSTM(500, return_sequences=True),
    tf.keras.layers.LSTM(500),
    tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Lambda(lambda x: x * 400)
])


# model.compile(optimizer='adam',
#               loss='mae',
#               metrics=['accuracy'])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(optimizer=optimizer,
              loss='mae',
              # loss=tf.keras.losses.Huber(),
              metrics=['accuracy'])

model.summary()
# history = model.fit(x_train, y_train, epochs=100, verbose=1, shuffle=False,
#                     validation_data=(x_test, y_test), callbacks=[lr_schedule])

log_dir = "logdir"   # + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(dataset, epochs=65, verbose=1, callbacks=[lr_schedule, tensorboard_callback])

# lrs = 1e-8 * (10 ** (np.arange(100) / 20))
# plt.semilogx(lrs, history.history['loss'])
# plt.axis([1e-8, 1e-3, 0, 300])
# loss = history.history['loss']
# epochs = range(0, len(loss))
# plot_loss = loss
# plt.plot(epochs, plot_loss, 'b', label='Training Loss')
#
# plt.show()

# Predicting using the model
# forecast = []
# results = []
# for i in range(len(series) - n_steps):
#     # forecast.append(model.predict(dataset[i, :-1].reshape((1, n_steps, n_features)))[0][0])
#     forecast.append(model.predict(series[i:i+n_steps, :-1].reshape((1, n_steps, n_features)).astype(float))[0][0])
#
# forecast = forecast[split_time-n_steps:]
# results = np.array(forecast)
#
# plt.plot(y_test)
# plt.plot(results)
# plt.show()

# print(tf.keras.metrics.mean_absolute_error(y_test, results).numpy())


# print(history.history)

# model.evaluate(X[16].reshape((1, n_steps, n_features)), y[16])

# y_hat = []
# for i in range(0, 17):
#     y_hat.append(model.predict(x_train[i].reshape((1, n_steps, n_features)))[0][0])

# plt.plot(y_hat, y_train, 'o')
# plt.show()

