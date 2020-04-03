import os
import logging
import tensorflow as tf
import numpy as np
import functools
from tensorflow import keras

FILTERS = 12
LATENT_DIMS = 100


def sampling(z_mean, z_logvar):
    batch, latent_dims = z_mean.shape
    epsilon = tf.random.normal(shape=z_mean.shape)

    return z_mean + np.exp(z_logvar/2)*epsilon


class Encoder(keras.layers.Layer):

    def __init__(self, num_output):
        super(Encoder, self).__init__()

        self.num_output = num_output
        self.Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
        self.BatchNormalization = tf.keras.layers.BatchNormalization
        self.Flatten = tf.keras.layers.Flatten
        self.Dense = functools.partial(tf.keras.layers.Dense, activation='relu')

    def call(self):

        model = tf.keras.Sequential([
            self.Conv2D(filters=1*FILTERS, kernel_size=5,  strides=2),
            self.BatchNormalization(),

            self.Conv2D(filters=2*FILTERS, kernel_size=5,  strides=2),
            self.BatchNormalization(),

            self.Conv2D(filters=4*FILTERS, kernel_size=3,  strides=2),
            self.BatchNormalization(),

            self.Conv2D(filters=6*FILTERS, kernel_size=3,  strides=2),
            self.BatchNormalization(),

            self.Flatten(),
            self.Dense(512),
            self.Dense(self.num_outputs, activation=None)])

        return model


class Decoder(keras.layers.Layers):

    def __init__(self,):
        super(Decoder, self).__init__()

        self.Conv2DTranspose = functools.partial(tf.keras.layers.Conv2DTranspose, padding='same', activation='relu')
        self.BatchNormalization = tf.keras.layers.BatchNormalization
        self.Flatten = tf.keras.layers.Flatten
        self.Dense = functools.partial(tf.keras.layers.Dense, activation='relu')
        self.Reshape = tf.keras.layers.Reshape

    def call(self):
        decoder = tf.keras.Sequential([
            # Transform to pre-convolutional generation
            self.Dense(units=4*4*6*FILTERS),  # 4x4 feature maps (with 6N occurances)
            self.Reshape(target_shape=(4, 4, 6*FILTERS)),

            # Upscaling convolutions (inverse of encoder)
            self.Conv2DTranspose(filters=4*FILTERS, kernel_size=3,  strides=2),
            self.Conv2DTranspose(filters=2*FILTERS, kernel_size=3,  strides=2),
            self.Conv2DTranspose(filters=1*FILTERS, kernel_size=5,  strides=2),
            self.Conv2DTranspose(filters=3, kernel_size=5,  strides=2)])

        return decoder


class DB_VAE(keras.models.Model):
    def __init__(self, latent_dims):
        super(DB_VAE, self).__init__()

        self.latent_dims = LATENT_DIMS
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        encoder_output = self.encoder(x)
        y_logit = tf.expend_dims(encoder_output[:, 0], -1)
        z_mean = encoder_output[:, 1:self.latent_dims+1]
        z_logvar = encoder_output[:, self.latent_dims+1:]

        return y_logit, z_mean, z_logvar

    def reparameterization(self, z_mean, z_logvar):
        z = sampling(z_mean, z_logvar)

        return z

    def decode(self, z):
        recon = self.decoder(z)

        return recon

    def call(self, inputs):
        y_logit, z_mean, z_logvar = self.encode(inputs)
        z = self.reparameterization(z_mean, z_logvar)
        recon = self.decode(z)

        return y_logit, z_mean, z_logvar, recon


def get_latent_mean(images, model, batch_size):
    '''
    Calculate mean of each batch
    Inputs : images(batch_size, H, W, C)
             model : DBVAE
             batch_size
    Outputs: mean's matrix: []
    '''
    num_imgs = images.shape[0]
    mean = np.zeros((num_imgs, LATENT_DIMS))
    for start in range(0, num_imgs, batch_size):
        end = min(start+batch_size, num_imgs+1)
        batch = (images[start:end]).astype(np.float32) / 255.

        _, batch_mean, _ = model.encode(batch)
        mean[start:end] = batch_mean

    return mean


def resampling(images, model, batch_size, bins=10, alpha=0.001):
    '''
    Resampling from latent distribution
    Inputs : images(batch_size, H, W, C)
             model : DBVAE
             bins (scalar)
             alpha (scalar): smoothing histogram
    '''
    mean = get_latent_mean(images, model, batch_size)
    training_sample_p = np.zeros_like(mean)

    for i in range(LATENT_DIMS):
        latent_distribution = mean[:, i]

        hist_density, edges = np.histogram(latent_distribution, bins=bins, density=True)

        edges[0] = -float('inf')
        edges[-1] = float('inf')

        bin_idx = np.digitize(latent_distribution, edges)

        smoothed_hist_density = hist_density + alpha
        smoothed_hist_density = smoothed_hist_density / np.sum(smoothed_hist_density)

        p = 1. / (smoothed_hist_density[bin_idx-1])
        p /= np.sum(p)

        training_sample_p = np.maximum(p, training_sample_p)

    training_sample_p /= np.sum(training_sample_p)

    return training_sample_p


def compute_vae_loss(x, recon, z_mean, z_logvar, kl_weight=0.0005):
    kl_loss = 1 + z_logvar - z_mean**2 - tf.exp(z_logvar)
    kl_loss = -0.5 * tf.reduce_mean(kl_loss, axis=1)

    recon_loss = tf.reduce_mean(tf.abs(x - recon), axis=(1, 2, 3))

    return recon_loss + kl_weight*kl_loss


@tf.function
def compute_dbvae_loss(x, recon, y, y_logit, z_mean, z_logvar):
    # loss = vae loss + classification loss if image has a face
    vae_loss = compute_vae_loss(x, recon, z_mean, z_logvar)

    classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(y, y_logit)

    face_indicator = tf.cast(tf.equal(y, 1), tf.float32)

    loss = tf.reduce_mean(vae_loss + classification_loss*face_indicator)

    return loss, classification_loss
