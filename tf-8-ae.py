import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

tf.get_logger().setLevel(logging.ERROR)

# hyperparams

image_size = 28*28
h_dim = 20
num_epochs = 55
batch_size = 100
learning_rate = 1e-3


class AutoEncoder(keras.Model):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.fc1 = keras.layers.Dense(512)
        self.fc2 = keras.layers.Dense(h_dim)

        self.fc3 = keras.layers.Dense(512)
        self.fc4 = keras.layers.Dense(image_size)

    def encode(self, x):
        x = tf.nn.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def decode(self, x):
        out = tf.nn.relu(self.fc3(x))
        out = self.fc4(x)

        return tf.nn.sigmoid(out)

    def call(self, inputs, training=None):
        # encode
        h = self.encode(inputs)
        # decode
        out = self.decode(h)

        return out


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))
train_db = tf.data.Dataset.from_tensor_slices((x_train))
train_db = train_db.shuffle(batch_size*5).batch(batch_size)

num_batch = x_train.shape[0] // batch_size  # 600 batchs, each batch consist of 100 imgs

model = AutoEncoder()
model.build(input_shape=(4, image_size))


model.compile(optimizer=keras.optimizers.Adam(0.001, clipnorm=15),
              loss=keras.losses.BinaryCrossentropy())

model.fit(x_train, x_train, epochs=50,
          batch_size=batch_size,
          shuffle=True,
          validation_data=(x_test, x_test))

predicted = model.predict(x_test)
num_display = 10
plt.figure(figsize=(10, 4))
for i in range(num_display):
    ax = plt.subplot(2, num_display, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, num_display, i+1+num_display)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
