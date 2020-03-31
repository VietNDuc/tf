import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, losses


(x_train, y_train), (x_val, y_val) = datasets.mnist.load_data()

x_train = tf.cast(x_train, tf.float32) / 255.
x_val = tf.cast(x_val, tf.float32) / 255.

y_train = tf.one_hot(tf.cast(y_train, tf.int64), depth=10)
y_val = tf.one_hot(tf.cast(y_val, tf.int64), depth=10)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_db = train_db.shuffle(60000).batch(100)
val_db = val_db.shuffle(10000).batch(100)


def run_gradient_tape():
    model = Sequential([
        layers.Reshape(target_shape=(28*28,), input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)])
    # model.build(input_shape=(None, 28*28))
    model.summary()

    optim = optimizers.Adam(lr=0.001)
    accu = metrics.CategoricalAccuracy()

    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.square(out - y)
            loss = tf.reduce_sum(loss) / 32

        accu.update_state(y, out)

        grads = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))

        if step % 200 == 0:
            print(step, 'Loss:', float(loss), '- Accu:', accu.result().numpy())
            accu.reset_states()


def run_model_keras():
    model = Sequential([
        layers.Reshape(target_shape=(28*28,), input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)])

    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_db.repeat(), epochs=30, steps_per_epoch=500,
              validation_data=val_db.repeat(),
              validation_steps=2)


# run_gradient_tape()
run_model_keras()
