import os

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback

import numpy as np
import random
import io
import sys


def main():
    file = sys.argv[1]
    fullPath = os.path.abspath("./" + file)
    print(fullPath)
    path = keras.utils.get_file(file, "file://" + fullPath)
    with io.open(path, encoding="utf-8") as f:
        text = f.read().lower()

    names = list(map(lambda s: s + '.', text.split("\n")))

    chars = sorted(list(set(text)))
    char_to_index = dict((c, i) for i, c in enumerate(chars))
    index_to_char = dict((i, c) for i, c in enumerate(chars))

    max_char = len(max(names, key=len))
    m = len(names)
    char_dim = len(chars)

    X = np.zeros((m, max_char, char_dim))
    Y = np.zeros((m, max_char, char_dim))

    for i in range(m):
        name = list(names[i])
        for j in range(len(name)):
            X[i, j, char_to_index[name[j]]] = 1
            if j < len(name) - 1:
                Y[i, j, char_to_index[name[j + 1]]] = 1

    # for i in range(len(X)):
    #     for j in range(len(X[i])):
    #         for k in range(len(X[i][j])):
    #             if X[i][j][k] != 0:
    #                 print("Not 0. i, j, k, value: " + str(i) + ", " + str(j) + ", " + str(k) + ", " + str(X[i][j][k]))
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_char, char_dim), return_sequences=True))
    model.add(Dense(char_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    def make_name(model):
        name = []
        x = np.zeros((1, max_char, char_dim))
        end = False
        i = 0

        while end == False:
            probs = list(model.predict(x)[0, i])
            probs = probs / np.sum(probs)
            index = np.random.choice(range(char_dim), p=probs)
            if i == max_char - 2:
                character = '.'
                end = True
            else:
                character = index_to_char[index]
            name.append(character)
            x[0, i + 1, index] = 1
            i += 1
            if character == '.':
                end = True

        print(''.join(name))

    def generate_name_loop(epoch, _):
        if epoch % 25 == 0:

            print('Names generated after epoch %d:' % epoch)

            for i in range(3):
                make_name(model)

            print()

    name_generator = LambdaCallback(on_epoch_end=generate_name_loop)
    model.fit(X, Y, batch_size=64, epochs=300, callbacks=[name_generator], verbose=1)
    for i in range(20):
        make_name(model)


if __name__ == "__main__":
    main()
