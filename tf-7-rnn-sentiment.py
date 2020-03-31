import os
import logging
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(22)


class RNN(keras.Model):

    def __init__(self, units, num_classes, num_layers):
        super(RNN, self).__init__()

        self.rnn = keras.layers.LSTM(units, return_sequences=True)
        self.rnn2 = keras.layers.LSTM(units)

        self.embedding = keras.layers.Embedding(top_words, 100, input_length=max_review_length)
        self.fc = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)

        x = self.rnn(x)
        x = self.rnn2(x)

        x = self.fc(x)

        return x


# load the dataset but only keep the top n words, zero the rest
top_words = 10000
# truncate and pad input sequences
max_review_length = 80
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)
# X_train = tf.convert_to_tensor(X_train)
# y_train = tf.one_hot(y_train, depth=2)
print('Pad sequences (samples x time)')
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length)
x_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


def main():
    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 20

    model = RNN(units, num_classes, num_layers=2)

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), verbose=1)

    scores = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print(scores)


if __name__ == '__main__':
    main()
