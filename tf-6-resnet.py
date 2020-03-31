import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(22)


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


def conv3x3(channels, strides=1, kernel=(3, 3)):
    return keras.layers.Conv2D(channels, kernel, strides=strides, padding='same',
                               use_bias=False,
                               kernel_initializer=tf.random_normal_initializer())


class ResnetBlock(keras.Model):
    # Full pre-activation : BN -> relu -> conv -> BN ->relu -> conv -> addition ->next_layer
    def __init__(self, channels, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.conv1 = conv3x3(channels, strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(channels, strides, kernel=(1, 1))
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self, input, training=None):
        residual = input

        x = self.bn1(input, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        if self.residual_path:
            residual = self.down_bn(input, training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)

        x = x + residual

        return x


class ResNet(keras.Model):

    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.num_blocks = len(block_list)
        self.block_list = block_list

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(self.out_channels)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')

        # build all the blocks
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):

                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_channels, strides=2, residual_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        block = ResnetBlock(self.out_channels, residual_path=True)
                    else:
                        block = ResnetBlock(self.out_channels, residual_path=False)

                self.in_channels = self.out_channels

                self.blocks.add(block)

            self.out_channels *= 2

        self.final_bn = keras.layers.BatchNormalization()
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):

        out = self.conv_initial(inputs)

        out = self.blocks(out, training=training)

        out = self.final_bn(out, training=training)
        out = tf.nn.relu(out)

        out = self.avg_pool(out)
        out = self.fc(out)

        return out


def main():
    train_db, val_db = mnist_data()
    epochs = 300
    num_classes = 10

    model = ResNet([2, 2, 2], num_classes)
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
