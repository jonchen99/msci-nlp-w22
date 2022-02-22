# Jonathan Chen, 20722167
# University of Waterloo
# March 4, 2022

import sys
import keras
import re
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def read_files(file_name):
    # Read data from files into a list
    modified_lines = []
    original_lines = []
    with open(file_name) as file:
        for line in file:
            # Removing the quotes and the newline character from the line
            string = re.sub("[\"\n]", "", line)
            # Adding a space after each period or comma
            string = string.replace('.', '. ', line.count('.')).replace(',', ', ', line.count(','))
            original_lines.append(string)
            modified_lines.append('<sos> ' + string + ' <eos>')
    
    return original_lines, modified_lines

def main(text_file, classifier):
    model = keras.models.load_model(f'data/nn_{classifier}.model')
    original_text, modified_text = read_files(text_file)

    # Read in tokenizer
    tokenizer_pkl = open('data/tokenizer.pkl', "rb")

    tokenizer, longest_sequence = pickle.load(tokenizer_pkl)
    text_sequence = tokenizer.texts_to_sequences(modified_text)

    text_pad = pad_sequences(text_sequence, maxlen=longest_sequence, padding='post')

    y_predict = model.predict(text_pad)
    y_classes = y_predict.argmax(axis=-1)

    for i, sentence in enumerate(original_text):
        print(f"SENTENCE: {sentence}\t LABEL: {y_classes[i]}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_text_file = sys.argv[1]
    classifier = sys.argv[2]

    # input_text_file = "/Users/jonathanchen/Documents/School/4B/MSCI 598/Assignments/Assignment4/sample_review.txt"
    # classifier = "tanh"

    main(input_text_file, classifier)
