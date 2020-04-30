import tensorflow as tf
import tensorflow.keras as keras
import logging
import numpy as np
from datetime import datetime

from somatictrainer.gestures import GestureTrainingSet, Gesture

logging.basicConfig(level=logging.DEBUG)

def main():
    corpus = GestureTrainingSet.load('E:\\Dropbox\\Projects\\Source-Controlled Projects\\Somatic\\Training Utility\\Corpus.json')

    data, labels = corpus.to_training_set()

    one_hot_labels = keras.utils.to_categorical(labels, num_classes=26)

    batch_size = 32

    model = keras.Sequential()
    model.add(keras.layers.LSTM(batch_size, activation='tanh', recurrent_activation='relu',
                                input_shape=data.shape[1:], return_sequences=True))
    model.add(keras.layers.Dropout(0.2))  # See https://stackoverflow.com/questions/48026129/how-to-build-a-keras-model-with-multidimensional-input-and-output

    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten(input_shape=data.shape[1:]))
    for i in range(5):
        model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dense(26, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(data, one_hot_labels, epochs=100, batch_size=batch_size, shuffle=True, verbose=True)

    model.evaluate(data, one_hot_labels, batch_size=batch_size)

    model.save('E:\\Dropbox\\Projects\\Source-Controlled Projects\\Somatic\\Training Utility\\{}.h5'.
               format(datetime.now().strftime('%Y-%M-%dT%H-%M')))

if __name__ == "__main__":
    main()
