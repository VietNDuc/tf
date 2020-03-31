import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow import keras

tf.random.set_seed(22)
tf.get_logger().setLevel(logging.ERROR)


def mnist_data():
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

    x_train = tf.cast(x_train, tf.float32) / 255.
    x_val = tf.cast(x_val, tf.float32) / 255.
    x_train, x_val = np.expand_dims(x_train, axis=3), np.expand_dims(x_val, axis=3)  # need rank=4

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_db = train_db.batch(500)
    val_db = val_db.batch(500)

    return train_db, val_db


def conv_bn_relu(channels, kernel=3, strides=1, padding='same'):
    model = keras.models.Sequential([
        keras.layers.Conv2D(channels, kernel, strides=strides, padding=padding),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU()
        ])

    return model


class InceptionBlock(keras.Model):
    def __init__(self, channels, strides=1):
        super(InceptionBlock, self).__init__()

        self.channel = channels
        self.strides = strides

        self.conv1 = conv_bn_relu(channels, strides=strides)
        self.conv2 = conv_bn_relu(channels, kernel=3, strides=strides)
        self.conv3_1 = conv_bn_relu(channels, kernel=3, strides=strides)
        self.conv3_2 = conv_bn_relu(channels, kernel=3, strides=1)

        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = conv_bn_relu(channels, strides=strides)

    def call(self, x, training=None):
        x1 = self.conv1(x, training=training)
        x2 = self.conv2(x, training=training)

        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)

        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)

        # concat along axis=channel
        x = tf.concat([x1, x2, x3_2, x4], axis=3)

        return x


class Inception(keras.Model):
    def __init__(self, num_layers, num_classes, in_channel=16, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.in_channel = in_channel
        self.out_channel = in_channel
        self.num_layers = num_layers
        self.init_ch = in_channel

        self.conv1 = conv_bn_relu(in_channel)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')
        for block_id in range(num_layers):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlock(self.out_channel, strides=2)
                else:
                    block = InceptionBlock(self.out_channel, strides=1)
                self.blocks.add(block)

            self.out_channel *= 2

        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, x, training=None):
        out = self.conv1(x, training=training)

        out = self.blocks(out, training=training)
        out = self.avg_pool(out)

        out = self.fc(out)

        return out


def main():
    train_db, val_db = mnist_data()
    epochs = 100
    model = Inception(2, 10)
    # derive input shape for every layers.
    model.build(input_shape=(None, 28, 28, 1))

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)

    acc_meter = keras.metrics.Accuracy()

    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # print(x.shape, y.shape)
                # [b, 10]
                logits = model(x)
                # [b] vs [b, 10]
                loss = criteon(tf.one_hot(y, depth=10), logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 10 == 0:
                print(epoch, step, 'loss:', loss.numpy())

        acc_meter.reset_states()
        for x, y in val_db:
            # [b, 10]
            logits = model(x, training=False)
            # [b, 10] => [b]
            pred = tf.argmax(logits, axis=1)
            # [b] vs [b, 10]
            acc_meter.update_state(y, pred)

        print(epoch, 'evaluation acc:', acc_meter.result().numpy())


if __name__ == '__main__':
    main()
