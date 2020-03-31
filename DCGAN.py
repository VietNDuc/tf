from tensorflow.keras import Model, Sequential, layers


class Generator(Model):
    def __init__(self):
        # hidden dims = 100
        super(Generator, self).__init__()

        self.model = Sequential()
        self.model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Reshape((7, 7, 256)))
        self.model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))  # (7, 7, 128)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))  # (14, 14, 64)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))  # (28, 28, 1) : image dims

    def call(self, input, training=None):
        return self.model(input)


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = Sequential()
        self.model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))

    def call(self, input, training=None):
        return self.model(input)
