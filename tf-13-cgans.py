import os
import logging

import matplotlib.pyplot as plt
import tensorflow as tf
from absl import app
from absl import flags

import cGANs.cGANs as cgans

tf.get_logger().setLevel(logging.ERROR)

FLAGS = flags.FLAGS
flags.DEFINE_integer('buffer_size', 400, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 1, 'Batch Size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer('lambda_value', 100, 'Value of lambda - hyperparam of MSE in generator loss')
flags.DEFINE_string('path', None, 'Path to the data folder')
flags.DEFINE_boolean('enable_function', True, 'Enable Function?')
flags.DEFINE_boolean('enable_tensorboard', True, 'Enable Tensorboard?')

IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
AUTOTUNE = tf.data.experimental.AUTOTUNE
_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

def preprocessing(image_file, training=None):
    # training dataset : resize(286, 286, 3) -> crop(256, 256, 3) -> randomly flip -> normalization
    # testing dataset : resize(256, 256, 3) -> normalization
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if training:
        # Resize to (286, 286, 3)
        input_image = tf.image.resize(input_image, [286, 286],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [286, 286],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Crop to (256, 256, 3)
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
        input_image, real_image = cropped_image[0], cropped_image[1]

        # Randomly flip
        if tf.random.uniform(()) > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

    else:
        input_image = tf.image.resize(input_image, size=[IMG_HEIGHT, IMG_WIDTH])
        real_image = tf.image.resize(real_image, size=[IMG_HEIGHT, IMG_WIDTH])

    # Normalization [-1, 1] : 2 * x / 255 - 1
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def get_dataset(path, buffer_size, batch_size):
    path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True, cache_dir=path)
    PATH_ds = os.path.join(os.path.dirname(path_to_zip), 'facades/')

    train_dataset = tf.data.Dataset.list_files(PATH_ds+'/train/*.jpg')
    train_dataset = train_dataset.map(lambda x: preprocessing(x, training=True),
                                      num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

    test_dataset = tf.data.Dataset.list_files(PATH_ds+'/train/*.jpg')
    test_dataset = test_dataset.map(lambda x: preprocessing(x, training=False),
                                    num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset


def generate_images(model, test_input, target):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def run_main(argv):
    del argv
    kwargs = {'epochs': FLAGS.epochs, 'enable_function': FLAGS.enable_function,
              'enable_tensorboard': FLAGS.enable_tensorboard,
              'path': FLAGS.path, 'buffer_size': FLAGS.buffer_size,
              'batch_size': FLAGS.batch_size, 'lambda_value': FLAGS.lambda_value}
    main(**kwargs)


def main(epochs, enable_function, enable_tensorboard, path, buffer_size, batch_size, lambda_value):
    model = cgans.pix2pix(epochs, enable_function, enable_tensorboard, lambda_value)

    train_dataset, _ = get_dataset(path, buffer_size, batch_size)
    checkpoint_pr = cgans.get_checkpoint_prefix()
    print('Training ...')

    return model.train(train_dataset, checkpoint_pr)


if __name__ == '__main__':
    app.run(run_main)
