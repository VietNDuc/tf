import os
import logging
import argparse
import tensorflow as tf
import numpy as np
import cv2
import functools
import h5py as h5
from tensorflow import keras
import matplotlib.pyplot as plt

from dbvae import *

tf.get_logger().setLevel(logging.ERROR)

PATH = 'https://www.dropbox.com/s/l5iqduhe0gwxumq/train_face.h5?dl=1'
PATH_DATA = keras.utils.get_file('train_face.h5', PATH)
BATCH_SIZE = 32
LR_RATE = 5e-4
LATENT_DIMS = 100
NUM_EPOCHS = 6


class Loader(object):
    def __init__(self, path):
        self.cache = h5.File(PATH_DATA, 'r')

        self.images = self.cache['images'][:]
        self.labels = self.cache['labels'][:].astype(np.float32)

        self.train_idx = np.random.permutation(np.arange(self.images.shape[0]))
        self.face_idx = self.train_idx[self.labels[self.train_idx, 0] == 1.]
        self.not_face_idx = self.train_idx[self.labels[self.train_idx, 0] != 1.]

    def get_train_size(self):
        return self.train_idx.shape[0]

    def get_all_faces(self):
        return self.images[self.face_idx]

    def get_batch(self, n, probs=None):
        selected_face_idx = np.random.choice(self.face_idx, size=n//2, replace=False, p=probs)
        selected_not_face_idx = np.random.choice(self.not_face_idx, size=n//2, replace=False)
        selected_idx = np.concatenate((selected_face_idx, selected_not_face_idx))

        train_images = (self.images[selected_idx]/255.).astype(np.float32)
        train_labels = self.labels[selected_idx]

        return (train_images, train_labels)


def main():
    loader = Loader(PATH_DATA)
    all_faces = loader.get_all_faces()

    model = DBVAE(LATENT_DIMS)
    optimizer = keras.optimizers.Adam(LR_RATE)

    for i in range(NUM_EPOCHS):
        print("Starting epoch {}/{} ".format(i+1, NUM_EPOCHS))

        probs_face = resampling(all_faces, model)

        for j in range(loader.get_train_size // BATCH_SIZE):

            x, y = loader.get_batch(BATCH_SIZE, probs=probs_face)
            with tf.GradientTape as tape:
                y_logit, z_mean, z_logvar, recon = model(x)

                total_loss, classification_loss = compute_dbvae_loss(x, recon, y, y_logit, z_mean, z_logvar)
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if j % 500 == 0:
                print('{} / {} -- total loss :'.format(j+1, loader.get_train_size, total_loss.numpy()))


if __name__ == '__main__':
    main()
