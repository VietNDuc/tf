import os
import logging
import argparse
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt


tf.get_logger().setLevel(logging.ERROR)

high_dims = 512
latent_dims = 2
epochs = 50
batch_size = 128

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
img_size = x_train.shape[1]
origin_dims = img_size * img_size
x_train = x_train.reshape((-1, origin_dims))
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
train_db = tf.data.Dataset.from_tensor_slices((x_train))
train_db = train_db.shuffle(batch_size*5).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test)).shuffle(batch_size*5).batch(batch_size)


class VAE(keras.Model):

    def __init__(self):
        super(VAE, self).__init__()
        #
        self.in_ = keras.layers.InputLayer(input_shape=(origin_dims,))
        self.fc1 = keras.layers.Dense(high_dims, activation='relu')
        self.mean_ = keras.layers.Dense(latent_dims)
        self.std_ = keras.layers.Dense(latent_dims)

        self.latent_inputs = keras.layers.InputLayer(input_shape=(latent_dims,))
        self.fc2 = keras.layers.Dense(high_dims, activation='relu')
        self.out_ = keras.layers.Dense(origin_dims, activation='sigmoid')

    def encode(self, x):
        z = self.in_(x)
        z = self.fc1(z)
        z_mean = self.mean_(z)
        z_log_var = self.std_(z)

        return (z_mean, z_log_var)

    def reparameterization(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=z_mean.shape)

        return z_mean + tf.exp(0.5*z_log_var) * epsilon

    def decode(self, z):
        z = self.latent_inputs(z)  # 128, 2
        z = self.fc2(z)
        recon = self.out_(z)

        return recon

    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterization(z_mean, z_log_var)
        recon = self.decode(z)

        kl_loss = 1 + z_log_var - np.square(z_mean) - np.exp(z_log_var)
        kl_loss = np.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        self.add_loss(kl_loss)

        return recon


@tf.function
def compute_loss(model, inputs, mse=True):
    z_mean, z_log_var = model.encode(inputs)
    z = model.reparameterization(z_mean, z_log_var)
    outputs = model.decode(z)

    if mse:
        reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    else:
        reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)

    # reconstruction_loss *= origin_dims
    recon_loss = tf.reduce_sum(reconstruction_loss)
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - z_mean**2 - tf.exp(z_log_var), axis=1)
    vae_loss = tf.reduce_mean(recon_loss + kl_loss)

    return vae_loss


def main():
    # parser = argparse.ArgumentParser()
    # help_ = "Use mse instead of binary ce"
    # parser.add_argument("-m",
    #                     "--mse",
    #                     help=help_, action='store_true')
    # help_ = "Load trained weights"
    # parser.add_argument("w",
    #                     "weights",
    #                     help=help_)
    # args = parser.parse_args()

    model = VAE()
    optimizer = keras.optimizers.Adam(0.001)
    mse_loss = tf.keras.losses.MeanAbsoluteError()

    metrics = tf.keras.metrics.Mean()

    for epoch in range(1, epochs+1):
        for step, x in enumerate(train_db):
            with tf.GradientTape() as tape:
                loss = compute_loss(model, x)

                # recon = model(x)
                # loss = mse_loss(x, recon)
                # loss += sum(model.losses)
                metrics(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 40 == 0:
                print(epoch, '---', step, '-- loss:', metrics.result().numpy())
                metrics.reset_states()

        # Generate image
        # z = tf.random.normal((batch_size, latent_dims))
        # out = model.decode(z)
        # out = tf.reshape(out, [-1, 28, 28]).numpy() * 255
        # out = out.astype(np.uint8)  # 100*28*28

        # np.savetxt('gen_img_array_{}.csv'.format(epoch), out, delimiter=',')


if __name__ == "__main__":
    main()
