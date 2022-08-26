import time
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from tensorflow import keras
import numpy as np
import os

step_length = 1  # The step length we take to get our samples from our corpus
epochs = 10  # Number of times we train on our full data
batch_size = 64  # Data samples in each training step
latent_dim = 256  # Size of our LSTM
dropout_rate = 0  # 0.2  # Regularization with dropout
model_path = os.path.realpath('model.h5')  # Location for the model
load_model = True  # Enable loading model from disk
generate_only = True  # Only generate names without training
store_model = False  # Store model to disk after training
verbosity = 1  # Print result for each epoch
gen_amount = 100  # How many

input_path = os.path.realpath('input.txt')
input_names = []

print('Reading names from file:')
with open(input_path, mode="r", encoding="utf-8") as f:
    for name in f:
        name = name.rstrip()
        if len(input_names) < 10:
            print(name)
        input_names.append(name)
    print('...')

concat_names = '\n'.join(input_names).lower()
chars = sorted(list(set(concat_names)))
print("Chars: " + str(chars))
num_chars = len(chars)
char2idx = dict((c, i) for i, c in enumerate(chars))
idx2char = dict((i, c) for i, c in enumerate(chars))
max_sequence_length = max([len(name) for name in input_names])

print('Total chars: {}'.format(num_chars))
print('Corpus length:', len(concat_names))
print('Number of names: ', len(input_names))
print('Longest name: ', max_sequence_length)

sequences = []
next_chars = []
for i in range(0, len(concat_names) - max_sequence_length, step_length):
    sequences.append(concat_names[i: i + max_sequence_length])
    next_chars.append(concat_names[i + max_sequence_length])

num_sequences = len(sequences)

print('Number of sequences:', num_sequences)
print('First 10 sequences and next chars:')
for i in range(10):
    print('X=[{}]   y=[{}]'.replace('\n', ' ').format(sequences[i], next_chars[i]).replace('\n', ' '))

X = np.zeros((num_sequences, max_sequence_length, num_chars), dtype=bool)
Y = np.zeros((num_sequences, num_chars), dtype=bool)

for i, sequence in enumerate(sequences):
    for j, char in enumerate(sequence):
        X[i, j, char2idx[char]] = 1
    Y[i, char2idx[next_chars[i]]] = 1

print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

model = Sequential()
model.add(LSTM(latent_dim, input_shape=(max_sequence_length, num_chars), recurrent_dropout=dropout_rate))
model.add(Dense(units=num_chars, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

def save_and_generate(epoch, _):
    if store_model:
        print('Storing model at:', model_path)
        model.save(model_path)
    print('Names generated after epoch %d:' % epoch)

    for i in range(3):
        print(generate_name(model))

def index_of(arr, str):
    try:
        return arr.index(str)
    except Exception as err:
        return -1

def generate_name(model):
    random_index = random.randint(0, len(concat_names) - 1 - max_sequence_length)
    start_index = concat_names.index("\n", random_index) + 1
    sequence = concat_names[start_index:start_index + max_sequence_length - 1] + '\n'
    while True:
        x = np.zeros((1, max_sequence_length, num_chars))
        for i, char in enumerate(sequence):
            x[0, i, char2idx[char]] = 1
        probs = model.predict(x, verbose=0)[0]
        next_idx = sample(probs)
        next_char = idx2char[next_idx]

        # New line means we have a new name
        if next_char == '\n':
            split = sequence.split('\n')
            gen_name = split[len(split) - 1]
            starts_w_duplicate = len(gen_name) > 2 and gen_name[0] == gen_name[1]
            has_two_words = index_of(gen_name, ' ') != -1
            is_too_short = len(gen_name) <= 2
            is_duplicate_name = gen_name in input_names
            is_too_long = len(gen_name) > 40
            if starts_w_duplicate or is_too_short or is_duplicate_name or is_too_long:
                return generate_name(model)  # start over

            if not has_two_words:
                sequence = sequence[1:] + " "
                continue

            split_name = gen_name.split(" ")
            final_name = ""
            for word in split_name:
                final_name += word.capitalize() + " "
            return final_name[:-1]
        sequence = sequence[1:] + next_char

def sample(preds, temperature=1.1):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if load_model:
    model = keras.models.load_model(model_path)
    print("Loaded model:")
    model.summary()

if not generate_only:
    start = time.time()
    print('Start training for {} epochs'.format(epochs))
    callback = LambdaCallback(on_epoch_end=save_and_generate)
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, callbacks=[callback], verbose=verbosity)
    end = time.time()
    print('Finished training - time elapsed:', (end - start) / 60, 'min')

if store_model and not generate_only:
    print('Storing model at:', model_path)
    model.save(model_path)

print("Generated names:")
for i in range(gen_amount):
    print(generate_name(model))
