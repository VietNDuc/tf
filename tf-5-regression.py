import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


class Regression(keras.layers.Layer):
    def __init__(self):
        super(Regression, self).__init__()
        self.w = self.add_variable('weight', [13, 1])
        self.b = self.add_variable('bias', [1])

        print(self.w.shape, self.b.shape)
        print(type(self.w), tf.is_tensor(self.w), self.w.name)
        print(type(self.b), tf.is_tensor(self.b), self.b.name)

    def call(self, x):
        x = tf.matmul(x, self.w) + self.b

        return x


def main():
    tf.random.set_seed(22)

    (x_train, y_train), (x_val, y_val) = keras.datasets.boston_housing.load_data()

    x_train, x_val = x_train.astype(np.float32), x_val.astype(np.float32)

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
    val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(102)

    model = Regression()
    criterion = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    losses_train = []
    losses_val = []
    for epoch in range(200):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x)  # [x, 1]
                logits = tf.squeeze(logits, axis=1)  # [x]
                loss = criterion(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses_train.append(loss.numpy())
        print(epoch, '--', loss.numpy())

        if epoch % 10 == 0:
            for x, y in val_db:
                logits = tf.squeeze(model(x), axis=1)
                loss = criterion(y, logits)
                losses_val.append(loss.numpy())
                print(epoch, 'val loss:', loss.numpy())

    print("All training's losses :", losses_train)
    print("All val's losses :", losses_val)


if __name__ == main():
    main()
