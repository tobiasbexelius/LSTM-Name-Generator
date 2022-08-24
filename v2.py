import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import os

step_length = 1    # The step length we take to get our samples from our corpus
epochs = 50       # Number of times we train on our full data
batch_size = 32    # Data samples in each training step
latent_dim = 64    # Size of our LSTM
dropout_rate = 0.2 # Regularization with dropout
model_path = os.path.realpath('./poke_gen_model.h5') # Location for the model
load_model = False # Enable loading model from disk
store_model = True # Store model to disk after training
verbosity = 1      # Print result for each epoch
gen_amount = 10    # How many

input_path = os.path.realpath('input.txt')

input_names = []

print('Reading names from file:')
with open(input_path) as f:
    for name in f:
        name = name.rstrip()
        if len(input_names) < 10:
            print(name)
        input_names.append(name)
    print('...')

# Make it all to a long string
concat_names = '\n'.join(input_names).lower()

# Find all unique characters by using set()
chars = sorted(list(set(concat_names)))
num_chars = len(chars)

# Build translation dictionaries, 'a' -> 0, 0 -> 'a'
char2idx = dict((c, i) for i, c in enumerate(chars))
idx2char = dict((i, c) for i, c in enumerate(chars))

# Use longest name length as our sequence window
max_sequence_length = max([len(name) for name in input_names])

print('Total chars: {}'.format(num_chars))
print('Corpus length:', len(concat_names))
print('Number of names: ', len(input_names))
print('Longest name: ', max_sequence_length)

sequences = []
next_chars = []

# Loop over our data and extract pairs of sequances and next chars
for i in range(0, len(concat_names) - max_sequence_length, step_length):
    sequences.append(concat_names[i: i + max_sequence_length])
    next_chars.append(concat_names[i + max_sequence_length])

num_sequences = len(sequences)

print('Number of sequences:', num_sequences)
print('First 10 sequences and next chars:')
for i in range(10):
    print('X=[{}]   y=[{}]'.replace('\n', ' ').format(sequences[i], next_chars[i]).replace('\n', ' '))

X = np.zeros((num_sequences, max_sequence_length, num_chars), dtype=np.bool)
Y = np.zeros((num_sequences, num_chars), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for j, char in enumerate(sequence):
        X[i, j, char2idx[char]] = 1
    Y[i, char2idx[next_chars[i]]] = 1

print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

model = Sequential()
model.add(LSTM(latent_dim,
               input_shape=(max_sequence_length, num_chars),
               recurrent_dropout=dropout_rate))
model.add(Dense(units=num_chars, activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)

model.summary()

if load_model:
    model.load_weights(model_path)
else:

    start = time.time()
    print('Start training for {} epochs'.format(epochs))
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbosity)
    end = time.time()
    print('Finished training - time elapsed:', (end - start) / 60, 'min')

if store_model:
    print('Storing model at:', model_path)
    model.save(model_path)

# Start sequence generation from end of the input sequence
sequence = concat_names[-(max_sequence_length - 1):] + '\n'

new_names = []

print('{} new names are being generated'.format(gen_amount))

while len(new_names) < gen_amount:

    # Vectorize sequence for prediction
    x = np.zeros((1, max_sequence_length, num_chars))
    for i, char in enumerate(sequence):
        x[0, i, char2idx[char]] = 1

    # Sample next char from predicted probabilities
    probs = model.predict(x, verbose=0)[0]
    probs /= probs.sum()
    next_idx = np.random.choice(len(probs), p=probs)
    next_char = idx2char[next_idx]
    sequence = sequence[1:] + next_char

    # New line means we have a new name
    if next_char == '\n':

        gen_name = [name for name in sequence.split('\n')][1]

        # Never start name with two identical chars, could probably also
        if len(gen_name) > 2 and gen_name[0] == gen_name[1]:
            gen_name = gen_name[1:]

        # Discard all names that are too short
        if len(gen_name) > 2:

            # Only allow new and unique names
            if gen_name not in input_names + new_names:
                new_names.append(gen_name.capitalize())

        if 0 == (len(new_names) % (gen_amount / 10)):
            print('Generated {}'.format(len(new_names)))

print_first_n = min(10, gen_amount)

print('First {} generated names:'.format(print_first_n))
for name in new_names[:print_first_n]:
    print(name)
