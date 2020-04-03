import logging

from absl import flags
from absl import app
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from cGANs.cGANs import Generator, Discriminator
from CycleGANs.CycleGANs import get_checkpoint_prefix, cycleGANs

FLAGS = flags.FLAGS
tf.get_logger().setLevel(logging.ERROR)
flags.DEFINE_integer('buffer_size', 1000, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 1, 'Batch Size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer('lambda_value', 10, 'Value of lambda - hyperparam of MSE in generator loss')
flags.DEFINE_boolean('enable_function', True, 'Enable Function?')
flags.DEFINE_boolean('enable_tensorboard', True, 'Enable Tensorboard?')
AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

raw_train_horse, raw_train_zebra = dataset['trainA'], dataset['trainB']
raw_test_horse, raw_test_zebra = dataset['testA'], dataset['testB']

def img_preprocess_train(image, label):
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, size=[256, 256, 3])

    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    image = (2*image / 255.) - 1

    return image

def img_preprocess_test(image, label):
    image = tf.cast(image, tf.float32)
    image = (2*image / 255.) - 1

    return image
print('Preprocessing dataset')
train_horse = raw_train_horse.map(img_preprocess_train, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(1)  # 1067
train_zebra = raw_train_zebra.map(img_preprocess_train, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(1)  # 1334
test_horse = raw_test_horse.map(img_preprocess_test, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(1)
test_zebra = raw_test_zebra.map(img_preprocess_test, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(1)

def run_main(argv):
    del argv
    kwargs = {'epochs': FLAGS.epochs, 'enable_function': FLAGS.enable_function,
              'enable_tensorboard': FLAGS.enable_tensorboard,
              'buffer_size': FLAGS.buffer_size,
              'batch_size': FLAGS.batch_size, 'lambda_value': FLAGS.lambda_value}
    main(**kwargs)


def main(epochs, enable_function, enable_tensorboard, buffer_size, batch_size, lambda_value):
    model = cycleGANs(epochs, enable_function, enable_tensorboard, lambda_value)

    # train_dataset, _ = get_dataset(path, buffer_size, batch_size)
    checkpoint_pr = get_checkpoint_prefix()
    print('Training ...')

    return model.train(train_horse, train_zebra, checkpoint_pr)


if __name__ == '__main__':
    app.run(run_main)
