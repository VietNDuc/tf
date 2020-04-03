import os
import datetime
import tqdm
import tensorflow as tf

OUTPUT_CHANNELS = 3

def get_checkpoint_prefix():
    checkpoint_dir = './cGANs/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    return checkpoint_prefix


def get_tensorboard():
    log_dir = "./cGANs/logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return summary_writer


class Downsample(tf.keras.Model):

    def __init__(self, filter, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.apply_batchnorm = apply_batchnorm
        self.conv2d = tf.keras.layers.Conv2D(filter, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)

        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()

    def call(self, inp, training=None):
        x = self.conv2d(inp)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = self.leaky_relu(x)

        return x


class Upsample(tf.keras.Model):

    def __init__(self, filter, size, apply_dropout=False):
        super(Upsample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.apply_dropout = apply_dropout

        self.conv2dT = tf.keras.layers.Conv2DTranspose(filter, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()

        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.3)
        self.relu = tf.keras.layers.ReLU()

    def call(self, input, training=None):
        x = self.conv2dT(input)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = self.relu(x)

        return x


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self.downsample_stack = [Downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
                                 Downsample(128, 4),  # (bs, 64, 64, 128)
                                 Downsample(256, 4),  # (bs, 32, 32, 256)
                                 Downsample(512, 4),  # (bs, 16, 16, 512)
                                 Downsample(512, 4),  # (bs, 8, 8, 512)
                                 Downsample(512, 4),  # (bs, 4, 4, 512)
                                 Downsample(512, 4),  # (bs, 2, 2, 512)
                                 Downsample(512, 4),  # (bs, 1, 1, 512)
                                 ]

        self.upsample_stack = [Upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
                               Upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
                               Upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
                               Upsample(512, 4),  # (bs, 16, 16, 1024)
                               Upsample(256, 4),  # (bs, 32, 32, 512)
                               Upsample(128, 4),  # (bs, 64, 64, 256)
                               Upsample(64, 4),  # (bs, 128, 128, 128)
                               ]
        self.inp = tf.keras.layers.InputLayer(input_shape=[None, None, 3])

        initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    activation='tanh')  # (bs, 256, 256, 3)

    def call(self, inp, training):
        skips = []
        x = self.inp(inp)
        for down in self.downsample_stack:
            x = down(x, training=training)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.upsample_stack, skips):
            x = up(x, training=training)
            x = tf.keras.layers.Concatenate()([x, skip])

        out = self.last(x)

        return out


class Discriminator(tf.keras.Model):
    def __init__(self, istarget):
        super(Discriminator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)
        self.istarget = istarget
        self.down1 = Downsample(64, 4, False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
        self.zero_pad = tf.keras.layers.ZeroPadding2D()
        self.conv2d = tf.keras.layers.Conv2D(512, 4, strides=1,
                                             kernel_initializer=initializer,
                                             use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                           kernel_initializer=initializer)

    def call(self, inputs, training):
        if self.istarget:
            inp, tar = inputs
            inputs = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
        x = self.down1(inputs, training=training)        # (bs, 128, 128, 64)
        x = self.down2(x, training=training)        # (bs, 64, 64, 128)
        x = self.down3(x, training=training)        # (bs, 32, 32, 256)

        x = self.zero_pad(x)                        # (bs, 34, 34, 256)
        x = self.conv2d(x)                          # (bs, 31, 31, 512)
        x = self.batchnorm(x, training=training)
        x = self.leaky_relu(x)

        x = self.zero_pad(x)                        # (bs, 33, 33, 512)
        x = self.last(x)                            # (bs, 30, 30, 1)

        return x


class pix2pix(object):
    def __init__(self, epochs, enable_function, enable_tensorboard, lambda_value):
        super(pix2pix, self).__init__()
        self.epochs = epochs
        self.enable_function = enable_function
        self.enable_tensorboard = enable_tensorboard
        self.lambda_value = lambda_value
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator = Generator()
        self.discriminator = Discriminator(istarget=True)
        if self.enable_tensorboard:
            self.summary_writer = get_tensorboard()
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optim,
            discriminator_optimizer=self.discriminator_optim,
            generator=self.generator,
            discriminator=self.discriminator)

    def generator_loss(self, generated_output, disc_generated_output, target_image):
        '''
        Compute generator's loss
        Input :
         - generated_output : Output of generator ~ G(x)
         - disc_generated_output : Output from discriminator ~ D(x, G(x))
        Output :
         - l1_loss : MAE between target image & generated_output ~ |y - G(x)|
         - gan_loss : Binary cross entropy between output of discriminator and array of 1s
                      ~ log(1 - D(x, G(x)))
        '''
        l1_loss = tf.reduce_mean(tf.abs(generated_output - target_image))
        gan_loss = self.cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)

        total_loss = gan_loss + self.lambda_value*l1_loss

        return total_loss, gan_loss, l1_loss

    def discriminator_loss(self, real_disc_output, disc_generated_output):
        '''
        Compute discriminator's loss
        Input :
         - real_disc_output : Output from discriminator ~ D(x, y)
         - disc_generated_output ~ D(x, G(x))
        Output:
         - real_loss: bce between real_disc_output and array of 1s ~ log(D(x, y))
         - fake_loss: bce between disc_generated_output and array of 0s ~ log(1 - D(x, G(x)))
        '''
        real_loss = self.cross_entropy(tf.ones_like(real_disc_output), real_disc_output)
        fake_loss = self.cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)

        return real_loss + fake_loss

    def train_step(self, epoch, input_image, target_image):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_output = self.generator(input_image, training=True)
            disc_generated_output = self.discriminator([input_image, generated_output], training=True)
            real_disc_output = self.discriminator([input_image, target_image], training=True)

            total_gen_loss, gan_loss, l1_loss = self.generator_loss(generated_output, disc_generated_output, target_image)
            disc_loss = self.discriminator_loss(real_disc_output, disc_generated_output)

        grad_of_gen = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        grad_of_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optim.apply_gradients(zip(grad_of_gen, self.generator.trainable_variables))
        self.discriminator_optim.apply_gradients(zip(grad_of_disc, self.discriminator.trainable_variables))

        if self.enable_tensorboard:
            with self.summary_writer.as_default():
                tf.summary.scalar('gen_total_loss', total_gen_loss, step=epoch)
                tf.summary.scalar('gen_gan_loss', gan_loss, step=epoch)
                tf.summary.scalar('gen_l1_loss', l1_loss, step=epoch)
                tf.summary.scalar('disc_loss', disc_loss, step=epoch)

        return total_gen_loss, disc_loss

    def train(self, train_dataset, checkpoint_prefix):

        if self.enable_function:
            self.train_step = tf.function(self.train_step)

        for epoch in range(self.epochs):
            print('Epoch {} / {}'.format(epoch+1, self.epochs))
            batch = tqdm.tqdm(total=400, desc='Batch')
            for (input_batch, target_batch) in train_dataset:
                gen_loss, disc_loss = self.train_step(epoch, input_batch, target_batch)
                batch.update(1)

            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix=checkpoint_prefix)

            print('Epoch {}, Generator loss {}, Discriminator Loss {}'.format(epoch, gen_loss, disc_loss))
