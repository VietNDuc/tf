import os
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, metrics


tf.get_logger().setLevel(logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_DIR = '/Users/vietnd/Documents/Saved_model/'
NUM_TRAIN_EPOCHS = 5


def mnist_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_set = train_ds.take(60000).shuffle(60000).batch(100)
    test_set = test_ds.batch(100)

    return train_set, test_set


model_dnn = keras.Sequential([
    layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10)])

model_cnn = keras.Sequential([
    layers.Reshape(target_shape=[28, 28, 1], input_shape=(28, 28, 1)),
    layers.Conv2D(2, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    layers.Conv2D(4, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dropout(rate=0.4),
    layers.Dense(10)])

optimizer_dnn = optimizers.Adam()
optimizer_cnn = optimizers.SGD(learning_rate=0.01, momentum=0.5)

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_step(model, optimizer, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = compute_loss(labels, logits)
        compute_accuracy(labels, logits)

    # compute grad
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train(model, optimizer, dataset, log_freq=50):
    avg_loss = metrics.Mean('loss', dtype=tf.float32)

    for images, labels in dataset:
        loss = train_step(model, optimizer, images, labels)
        avg_loss(loss)

        if tf.equal(optimizer.iterations % log_freq, 0):
            print('Step:', int(optimizer.iterations),
                  'loss:', avg_loss.result().numpy(),
                  'accu', compute_accuracy.result().numpy())
            avg_loss.reset_states()
            compute_accuracy.reset_states()


def test(model, dataset, step_num):
    avg_loss = metrics.Mean('loss', dtype=tf.float32)
    for images, labels in dataset:
        logits = model(images, training=False)
        avg_loss(compute_loss(labels, logits))
        compute_accuracy(labels, logits)

    print("Test set loss: {:0.4f} - accu: {:0.2f}%".format(
        avg_loss.result(),
        compute_accuracy.result()*100))
    print('Loss:', avg_loss.result(), '- Accu:', compute_accuracy.result())


def apply_clean():
    if tf.io.gfile.exists(MODEL_DIR):
        print('Removing existing model dir:', MODEL_DIR)
        tf.io.gfile.rmtree(MODEL_DIR)


apply_clean()

checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

checkpoint = tf.train.Checkpoint(model=model_cnn, optimizer=optimizer_cnn)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

train_set, test_set = mnist_dataset()
for i in range(NUM_TRAIN_EPOCHS):
    start = time.time()
    train(model_cnn, optimizer_cnn, train_set, log_freq=500)
    end = time.time()

    print('Train time for epoch #{} ({} total steps): {}'.format(
        i + 1, int(optimizer_cnn.iterations), end - start))
    checkpoint.save(checkpoint_prefix)
    print('Saved checkpoint!')

export_path = os.path.join(MODEL_DIR, 'export')
tf.saved_model.save(model_cnn, export_path)
print('saved SavedModel for exporting')
