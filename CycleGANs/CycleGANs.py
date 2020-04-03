import os
import time
import datetime
import tqdm

import tensorflow as tf
from cGANs.cGANs import Discriminator, Generator

def get_checkpoint_prefix():
    checkpoint_dir = './CycleGANs/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    return checkpoint_prefix

def get_tensorboard():
    log_dir = "./CycleGANs/logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return summary_writer


class cycleGANs(object):

    def __init__(self, epochs, lambda_value, enable_function, enable_tensorboard):
        super(cycleGANs, self).__init__()

        self.epochs = epochs
        self.lambda_value = lambda_value
        self.enable_function = enable_function
        self.enable_tensorboard = enable_tensorboard
        self.generator_x2y = Generator()
        self.generator_y2x = Generator()

        self.discriminator_x = Discriminator(istarget=False)
        self.discriminator_y = Discriminator(istarget=False)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_x2y_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_y2x_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        if self.enable_tensorboard:
            self.summary_writer = get_tensorboard()
        self.checkpoint = tf.train.Checkpoint(
            generator_x2y_optim=self.generator_x2y_optim,
            generator_y2x_optim=self.generator_y2x_optim,
            discriminator_x_optim=self.discriminator_x_optim,
            discriminator_y_optim=self.discriminator_y_optim,
            generator_x2y=self.generator_x2y,
            generator_y2x=self.generator_y2x,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y)

    def generator_loss(self, generated):
        #
        return self.cross_entropy(tf.ones_like(generated), generated)

    def discriminator_loss(self, real, fake):
        #
        real_loss = self.cross_entropy(tf.ones_like(real), real)
        fake_loss = self.cross_entropy(tf.zeros_like(fake), fake)
        total_loss = real_loss + fake_loss

        return 0.5 * total_loss

    def identity_loss(self, real, same):
        #
        loss = tf.reduce_mean(tf.abs(real - same))

        return self.lambda_value*loss*0.5

    def cycle_loss(self, real, cycle):
        #
        loss = tf.reduce_mean(tf.abs(real - cycle))

        return self.lambda_value*loss

    def train_step(self, epoch, x, y):
        # Train each step with GradientTape()
        with tf.GradientTape(persistent=True) as tape:
            generated_y = self.generator_x2y(x, training=True)
            generated_x = self.generator_y2x(y, training=True)

            same_y = self.generator_x2y(y, training=True)
            same_x = self.generator_y2x(x, training=True)

            cycle_y = self.generator_x2y(generated_x, training=True)
            cycle_x = self.generator_y2x(generated_y, training=True)

            disc_gen_x = self.discriminator_x(generated_x, training=True)
            disc_gen_y = self.discriminator_y(generated_y, training=True)

            disc_real_x = self.discriminator_x(x, training=True)
            disc_real_y = self.discriminator_y(y, training=True)

            gen_loss_x2y = self.generator_loss(disc_gen_y)
            gen_loss_y2x = self.generator_loss(disc_gen_x)

            cycle_loss = self.cycle_loss(x, cycle_x) + self.cycle_loss(y, cycle_y)

            total_gen_loss_x2y = gen_loss_x2y + cycle_loss + self.identity_loss(y, same_y)
            total_gen_loss_y2x = gen_loss_y2x + cycle_loss + self.identity_loss(x, same_x)

            disc_loss_x = self.discriminator_loss(disc_real_x, disc_gen_x)
            disc_loss_y = self.discriminator_loss(disc_real_y, disc_gen_y)
        grad_of_gen_x2y = tape.gradient(total_gen_loss_x2y, self.generator_x2y.trainable_variables)
        grad_of_gen_y2x = tape.gradient(total_gen_loss_y2x, self.generator_y2x.trainable_variables)
        grad_of_disc_y = tape.gradient(disc_loss_y, self.discriminator_y.trainable_variables)
        grad_of_disc_x = tape.gradient(disc_loss_x, self.discriminator_x.trainable_variables)

        self.generator_x2y_optim.apply_gradients(zip(grad_of_gen_x2y, self.generator_x2y.trainable_variables))
        self.generator_y2x_optim.apply_gradients(zip(grad_of_gen_y2x, self.generator_y2x.trainable_variables))
        self.discriminator_x_optim.apply_gradients(zip(grad_of_disc_x, self.discriminator_x.trainable_variables))
        self.discriminator_y_optim.apply_gradients(zip(grad_of_disc_y, self.discriminator_y.trainable_variables))

        if self.enable_tensorboard:
            with self.summary_writer.as_default():
                tf.summary.scalar('Generator loss x2y', total_gen_loss_x2y, step=epoch)
                tf.summary.scalar('Generator loss y2x', total_gen_loss_y2x, step=epoch)
                tf.summary.scalar('Discriminator loss x', disc_loss_x, step=epoch)
                tf.summary.scalar('Discriminator loss y', disc_loss_y, step=epoch)

        return total_gen_loss_x2y + total_gen_loss_y2x, disc_loss_x + disc_loss_y

    def train(self, train_dataset_1, train_dataset_2, checkpoint_prefix):

        if self.enable_function:
            self.train_step = tf.function(self.train_step)

        for epoch in range(self.epochs):
            print('Epoch {} / {}'.format(epoch+1, self.epochs))
            start = time.time()
            batch = tqdm.tqdm(total=1067, desc='Batch')
            for (inp1, inp2) in tf.data.Dataset.zip((train_dataset_1, train_dataset_2)):
                gen_loss, disc_loss = self.train_step(epoch, inp1, inp2)
                batch.update(1)

            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix=checkpoint_prefix)
            print('Time taken for epoch {} is {}\n'.format(epoch+1, time.time() - start))
            print('Epoch {}, Generator loss {}, Discriminator Loss {}'.format(epoch, gen_loss, disc_loss))

