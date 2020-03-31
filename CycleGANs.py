from absl import flags
import tensorflow as tf
import cGANs as cgans

flags = flags.FLAGS


class cycleGANs(object):

    def __init__(self):
        super(cycleGANs, self).__init__()

        self.generator_x2y = cgans.Generator()
        self.generator_y2x = cgans.Generator()

        self.dicriminator_x = cgans.Discriminator()
        self.discriminator_y = cgans.Discriminator()
        self.cross_entropy = tf.keras.losses.Binartcrossentropy(from_logits=True)

    def generator_loss(self, generated):
        return self.cross_entropy(tf.ones_like(generated), generated)

    def discriminator_loss(self, real, fake):
        real_loss = self.cross_entropy(tf.ones_like(real), real)
        fake_loss = self.cross_entropy(tf.zeros_like(fake), fake)

        return real_loss + fake_loss

    def identity_loss(self, real, same):
        return tf.reduce_mean(tf.abs(real - same))

    def cycle_loss(self, real, cycle):
        return tf.reduce_mean(tf.abs(real - cycle))

    def train_step(self):
