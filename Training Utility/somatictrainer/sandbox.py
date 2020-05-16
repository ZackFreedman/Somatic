import random
import time

import requests
import tensorflow.keras as keras
import tensorflow as tf
import sklearn.model_selection
import logging
import numpy as np
import hexdump
from datetime import datetime

from somatictrainer.gestures import GestureTrainingSet, Gesture

logging.basicConfig(level=logging.DEBUG)


class Callbacks(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        logging.info('Starting training!')

    def on_train_end(self, logs=None):
        logging.info('Training over!')

    def on_train_batch_begin(self, batch, logs=None):
        logging.info('Starting batch {}. Logs: {}'.format(batch, logs))

    def on_train_batch_end(self, batch, logs=None):
        logging.info('Ending batch {}. Logs: {}'.format(batch, logs))

    def on_epoch_begin(self, epoch, logs=None):
        logging.info('Starting epoch {}'.format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        logging.info('Ending epoch {}. Logs: {}'.format(
            epoch, logs))


def make_model():
    filename = 'training_set_2.db'

    corpus = GestureTrainingSet.load(
        'E:\\Dropbox\\Projects\\Source-Controlled Projects\\Somatic\\Training Utility\\' + filename)

    data, labels = corpus.to_training_set()

    one_hot_labels = keras.utils.to_categorical(labels, num_classes=np.max(labels) + 1)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data, one_hot_labels, test_size=0.1, shuffle=True, stratify=one_hot_labels)

    batch_size = 32

    model = keras.Sequential()
    # model.add(keras.layers.LSTM(batch_size, activation='sigmoid', recurrent_activation='relu',
    #                             input_shape=data.shape[1:], return_sequences=True))
    # model.add(keras.layers.Dropout(0.2))  # See https://stackoverflow.com/questions/48026129/how-to-build-a-keras-model-with-multidimensional-input-and-output
    #
    # model.add(keras.layers.LSTM(64))
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten(input_shape=data.shape[1:]))
    model.add(keras.layers.Dense(batch_size, activation='relu'))
    for i in range(4):
        model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(np.max(labels) + 1, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, epochs=500, batch_size=batch_size,
              shuffle=True, verbose=True, validation_data=(x_test, y_test))

    model.evaluate(data, one_hot_labels, batch_size=batch_size)

    if '.' in filename:
        filename = filename[:filename.rindex('.')]

    model_name = 'E:\\Dropbox\\Projects\\Source-Controlled Projects\\Somatic\\Training Utility\\' + filename + '.h5'

    logging.info('Done! Model saved to ' + model_name)

    model.save(model_name)

    # model = keras.models.load_model('E:\\Dropbox\\Projects\\Source-Controlled Projects\\Somatic\\Training Utility\\' + filename + '.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter._experimental_new_quantizer = True
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.allow_custom_ops = True

    def rep_data_gen():
        for val in x_test:
            yield [np.array(val, dtype=np.float32, ndmin=2)]

    converter.representative_dataset = rep_data_gen

    tflite_model = converter.convert()
    open(filename + '.tflite', 'wb').write(tflite_model)

    hex_lines = []
    for chunk in hexdump.chunks(tflite_model, 16):
        hex_lines.append('0x' + hexdump.dump(chunk, sep=', 0x'))

    char_map = corpus.get_character_map()

    with open(filename + '_model.h', 'w') as f:
        f.write('const unsigned char modelBin[] = {\n')
        for index, line in enumerate(hex_lines):
            f.write('  ' + line)
            if index < len(hex_lines) - 1:
                f.write(',')
            f.write('\n')
        f.write('};\n\n')

        f.write('const unsigned int modelLen = {};\n\n'.format(len(tflite_model)))

        f.write('const byte charMap[] = {\n')
        highest_key = np.max(list(char_map.keys()))
        for i in range(highest_key + 1):
            if i % 10 == 0:
                f.write('  ')

            if i in char_map:
                f.write(hex(ord(char_map[i])))
            else:
                f.write('0x00')

            if i < highest_key:
                f.write(', ')
                if i % 10 == 9:
                    f.write('\n')
        f.write('\n};\n\n')

        f.write('#define charCount {}\n'.format(len(char_map)))


def generate_training_sentence():
    hipsum = requests.get('https://hipsum.co/api/?type=hipster-centric&sentences=3').json()[
        0]  # Get Hipster Ipsum, strip off trailing punctuation
    tokens = hipsum.split(' ')

    for i in range(len(tokens)):
        if random.randint(0, 2) == 0:
            # Randomly capitalize some of the words to collect more caps samples
            tokens[i] = tokens[i].capitalize()
        elif random.randint(0, 5) == 0:
            tokens[i] = tokens[i].upper()
        if i < len(tokens) - 1:
            tokens[i] += ' '

    number = ''

    for i in range(len(tokens)):
        if random.randint(0, 4) == 0:
            # Also add some numbers, we need more of those too
            for j in range(2, 6):
                number += chr(ord('0') + random.randint(0, 9))
            tokens.insert(i, number + ' ')

    symbols = '!"#$\',-./?@'  # No space - we have plenty 'o them

    for i in range(len(tokens)):
        if random.randint(0, 4) == 0:
            if i > 0:
                tokens[i - 1] = tokens[i - 1][:-1]  # Get rid of that space
            # Add some punctuation symbols too
            tokens.insert(i, symbols[random.randint(0, len(symbols) - 1)] + ' ')

    return ''.join(tokens)


if __name__ == "__main__":
    # while True:
    #     print(generate_training_sentence())
    #     time.sleep(1)
    make_model()
