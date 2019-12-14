from dataset import CrispDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
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
    return np.array(X, dtype=float), np.array(y, dtype=float)


class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


expert_data_path = "C:/Users/mohaddesi.s/Documents/PycharmProjects/MyFirstProgram/crisp_game_server" \
                   "/gamette_experiments/study_1/player_state_actions/"

dataset = CrispDataset(expert_data_path)
order = dataset.order
inventory = dataset.inventory
demand = dataset.demand
backlog = dataset.backlog
shipment = dataset.shipment

dataset = pd.DataFrame(columns=['inventory', 'demand', 'backlog', 'shipment', 'order'])
for i in range(0, 40):
    dataset = dataset.append(pd.concat([inventory.iloc[i, 0:20], demand.iloc[i, 0:20],
                                        backlog.iloc[i, 0:20], shipment.iloc[i, 0:20], order.iloc[i, 0:20]],
                                       axis=1,
                                       keys=['inventory', 'demand', 'backlog', 'shipment', 'order']))

dataset = dataset.to_numpy()

n_steps = 4
X, y = split_sequences(dataset, n_steps)
n_features = X.shape[2]

print(X.shape, y.shape)

# summarize the data
for i in range(len(X)):
    print(X[i], y[i])

callbacks = MyCallBack()

# plt.plot(np.array(range(0, 797)), y)
# plt.show()

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
model.summary()
history = model.fit(X, y, epochs=1000, verbose=0, callbacks=[callbacks])

print(history.history)

# model.evaluate(X[16].reshape((1, n_steps, n_features)), y[16])

y_hat = []
for i in range(0, 797):
    y_hat.append(model.predict(X[i].reshape((1, n_steps, n_features)))[0][0])

plt.plot(y_hat, y, 'o')
plt.show()
