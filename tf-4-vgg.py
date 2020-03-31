import os
import logging
import platform
import pickle
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, metrics, optimizers

from vgg16 import VGG16

tf.get_logger().setLevel(logging.ERROR)

DATASET_DIR = r'/Users/vietnd/Documents/Datasets/cifar-10-batches-py/'


def normalize(x):
    x = x / 255.
    mean = np.mean(x, axis=(0, 1, 2, 3))
    std = np.std(x, axis=(0, 1, 2, 3))
    x = (x - mean) / (std + 1e-7)

    return x


def prepare_cifar(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    return x, y


def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def load_pickle(file):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(file)
    elif version[0] == '3':
        return pickle.load(file, encoding='latin1')
    raise ValueError('Invalid pthon version: {}'.format(version))


def load_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        x = datadict['data']
        y = datadict['labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        y = np.array(y)
        return x, y


def load_all(ROOT):
    Xs = []
    Ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        x, y = load_batch(f)
        Xs.append(x)
        Ys.append(y)
    x_train = np.concatenate(Xs)
    y_train = np.concatenate(Ys)
    del x, y
    x_test, y_test = load_batch(os.path.join(ROOT, 'test_batch'))

    return x_train, y_train, x_test, y_test


def main():
    tf.random.set_seed(22)

    print('Loading data...')
    x, y, x_test, y_test = load_all(DATASET_DIR)
    x = normalize(x)
    x_test = normalize(x_test)

    train_loader = tf.data.Dataset.from_tensor_slices((x, y))
    train_loader = train_loader.map(prepare_cifar).shuffle(7000).batch(256)

    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.map(prepare_cifar).shuffle(3000).batch(256)
    print('Done!')

    model = VGG16([32, 32, 3])

    # must specify from_logits=True!
    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.CategoricalAccuracy()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    for epoch in range(250):

        for step, (x, y) in enumerate(train_loader):
            # [b, 1] => [b]
            # y = tf.squeeze(y)
            # [b, 10]
            y = tf.one_hot(tf.cast(y, np.int64), depth=10)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y, logits)

                metric.update_state(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)

            grads = [tf.clip_by_norm(g, 15) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 40 == 0:
                # for g in grads:
                #     print(tf.norm(g).numpy())
                print(epoch, step, 'loss:', float(loss), 'acc:', metric.result().numpy())
                metric.reset_states()

        if epoch % 1 == 0:

            metric = keras.metrics.CategoricalAccuracy()
            for x, y in test_loader:
                # [b, 1] => [b]
                y = tf.squeeze(y, axis=1)
                # [b, 10]
                y = tf.one_hot(tf.cast(y, np.int64), depth=10)

                logits = model.predict(x)
                # be careful, these functions can accept y as [b] without warnning.
                metric.update_state(y, logits)
            print('test acc:', metric.result().numpy())
            metric.reset_states()


if __name__ == '__main__':
    main()
