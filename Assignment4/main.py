# Jonathan Chen, 20722167
# University of Waterloo
# March 4, 2022

import re 
import os
import sys
import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, Dropout
from gensim.models import Word2Vec
import numpy as np
from keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle

EMBEDDING_DIMENSIONS = 200
L2_REGULARIZATION = 0.001
BATCH_SIZE = 1000

def read_files(file_name, isLabel):
    # Read data from files into a list
    lines = []
    with open(file_name) as file:
        for line in file:
            # Removing the quotes and the newline character from the line
            string = re.sub("[\"\n]", "", line)
            # Adding a space after each period or comma
            string = string.replace('.', '. ', line.count('.')).replace(',', ', ', line.count(','))
            if isLabel:
                lines.append(int(string))
            else:
                lines.append('<sos> ' + string + ' <eos>')
    
    return lines

def create_embedding_matrix(vocab, w2v):

    embedding_dim = EMBEDDING_DIMENSIONS
    # Add one because index 0 is reserved and isn't assigned to any word
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
    embedding_matrix = np.zeros((len(vocab)+1, embedding_dim))

    embedding_matrix[0] = np.random.random((1, embedding_dim))
    for i, word in enumerate(vocab, 1):
        try:
            embedding_matrix[i] = w2v.wv[word]
        except KeyError as e:
            embedding_matrix[i] = np.random.random((1, embedding_dim))

    return embedding_matrix

def main(file_path):

    if not os.path.exists("data"):
        os.makedirs("data")

    # Read in the input files
    train = read_files(file_path+"/train.csv", False)
    val = read_files(file_path + "/val.csv", False)
    test = read_files(file_path+"/test.csv", False)
    
    # train = read_files(file_path+"/train_ns.csv", False)
    # val = read_files(file_path + "/val_ns.csv", False)
    # test = read_files(file_path+"/test_ns.csv", False)

    train_labels = read_files(file_path+"/train_labels.csv", True)
    val_labels = read_files(file_path+"/val_labels.csv", True)
    test_labels = read_files(file_path+"/test_labels.csv", True)

    # Reducing the size of the dataset for testing
    # train = train[:1000]
    # val = val[:1000] 
    # test = test[:1000]
    # train_labels = train_labels[:1000]
    # val_labels = val_labels[:1000] 
    # test_labels = test_labels[:1000]

    # Convert to class matrix so labels are the right type and it doesn't error
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
    train_labels = to_categorical(np.array(train_labels))
    val_labels = to_categorical(np.array(val_labels))
    test_labels = to_categorical(np.array(test_labels))

    # Vectorizing the corpus
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train)
    train_sequence = tokenizer.texts_to_sequences(train)
    val_sequence = tokenizer.texts_to_sequences(val)
    test_sequence = tokenizer.texts_to_sequences(test)

    longest_sequences = [len(x) for x in (train_sequence + val_sequence)]
    longest_sequence = max(longest_sequences)

    # Save the tokenizer to a pickle
    tokenizer_pkl = open("data/tokenizer.pkl", "wb")
    pickle.dump([tokenizer, longest_sequence], tokenizer_pkl)
    tokenizer_pkl.close()

    # Pad all train, val, and test sentences to be the longest sequence length in the entire dataset (excluding test)
    train_pad = pad_sequences(train_sequence, maxlen=longest_sequence, padding='post')
    val_pad = pad_sequences(val_sequence, maxlen=longest_sequence, padding='post')
    test_pad = pad_sequences(test_sequence, maxlen=longest_sequence, padding='post')

    w2v = Word2Vec.load(f"../Assignment3/data/w2v.model")
    embedding_matrix = create_embedding_matrix(tokenizer.word_index.keys(), w2v)

    activation_functions = ['relu', 'sigmoid', 'tanh']
    
    # Suggests dropout values between 20%-50% https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    dropouts = [0.2, 0.3, 0.4, 0.5]

    for activation in activation_functions:
        best_model = Sequential()
        best_accuracy = 0

        for dropout in dropouts:
            model = Sequential()
            model.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                                output_dim=EMBEDDING_DIMENSIONS,
                                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                input_length=longest_sequence,
                                trainable=False,
                                name='embedding_layer',
                                ))
            
            model.add(Dense(EMBEDDING_DIMENSIONS, activation=activation, kernel_regularizer= l2(L2_REGULARIZATION)))

            model.add(Flatten())

            model.add(Dropout(dropout))

            model.add(Dense(2, activation='softmax', kernel_regularizer=l2(L2_REGULARIZATION), name='output'))
            # print(model.summary())

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

            model.fit(train_pad, train_labels, batch_size=BATCH_SIZE, epochs=10, validation_data=(val_pad, val_labels))
            
            _, accuracy = model.evaluate(test_pad, test_labels, batch_size=BATCH_SIZE)
            print(f"For activation function = {activation} and dropout={dropout}:")
            print("Test Set Accuracy = {:.4f}".format(accuracy))

            if (accuracy > best_accuracy):
                best_accuracy = accuracy
                best_model = model

        best_model.save(f'data/nn_{activation}.model')
        print(f"Model saved to {os.getcwd()}/data/nn_{activation}.model")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_folder_path = sys.argv[1]

    main(input_folder_path)
