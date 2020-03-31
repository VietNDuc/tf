#!/usr/bin/python

import os
import logging
import time
import glob
import tqdm
import imageio
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import losses, optimizers
from DCGAN import Generator, Discriminator

tf.get_logger().setLevel(logging.ERROR)
BATCH_SIZE = 256
BUFFER_SIZE = 60000
EPOCHS = 50
LR_RATE = 1e-4
HIDDEN_DIMS = 100
NUM_OF_EXAMPLES = 16


def get_data(BUFFER_SIZE, BATCH_SIZE):
    # Get data + convert to float32 + rescale to [-1, 1] + batch
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
    train_images = 2 * train_images / 255. - 1  # rescale to -1, 1
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    return train_dataset


def save_images(generator, input, epoch):
    # Generated 16 images from model and save it after each epoch
    generated_images = generator(input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def create_gif(to_gifname):
    with imageio.get_writer(to_gifname, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


cross_entropy = losses.BinaryCrossentropy(from_logits=True)


def generator_loss(generator, discriminator, input_noise, training):
    generated_img = generator(input_noise, training=training)
    fake_output = discriminator(generated_img, training=training)

    loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return loss


def discriminator_loss(generator, discriminator, real_image, input_noise, training):
    generated_img = generator(input_noise, training=training)
    fake_output = discriminator(generated_img, training=training)
    real_output = discriminator(real_image, training=training)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss


def main():
    seed = tf.random.normal([NUM_OF_EXAMPLES, HIDDEN_DIMS])
    train_dataset = get_data(BUFFER_SIZE, BATCH_SIZE)
    generator = Generator()
    discriminator = Discriminator()

    generator_optim = optimizers.Adam(learning_rate=LR_RATE)
    discriminator_optim = optimizers.Adam(learning_rate=LR_RATE)

    checkpoint_dir = './dcgan_training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optim,
                                     discriminator_optimizer=discriminator_optim,
                                     generator=generator,
                                     discriminator=discriminator)

    print('Start')
    for epoch in range(EPOCHS):
        print('Epoch {} / {}'.format(epoch+1, EPOCHS))
        start = time.time()
        for train_batch in train_dataset:
            input_noise = tf.random.normal([BATCH_SIZE, HIDDEN_DIMS])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
                gen_loss = generator_loss(generator, discriminator, input_noise, training=True)
                dis_loss = discriminator_loss(generator, discriminator, train_batch, input_noise, training=True)

            gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradient_dis = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
            generator_optim.apply_gradients(zip(gradient_gen, generator.trainable_variables))
            discriminator_optim.apply_gradients(zip(gradient_dis, discriminator.trainable_variables))

        save_images(generator, seed, epoch)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        stop = time.time()
        print('Take {} times to run epoch {}'.format(stop-start, epoch+1))

    print('Done!')
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


if __name__ == '__main__':
    main()
