# Jonathan Chen, 20722167
# University of Waterloo
# February 11, 2022

import pickle
import re
import sys
from nltk.corpus import stopwords
import nltk.tokenize as tok

def read_files(file_name, model):
    # Read data from files into a list
    original_lines = []
    tokenized_lines = []
    stop_words = set(stopwords.words('english'))

    with open(file_name) as file:
        for line in file:
            tokens = tok.word_tokenize(line)

            if 'ns' in model:
                tokens = [x for x in tokens if x.lower() not in stop_words]

            string = ", ".join(tokens)
            original_lines.append(re.sub("[\"\n]", "", line))
            tokenized_lines.append(string)
    
    return original_lines, tokenized_lines

def main(text_file, model):
    # Assumes the pickle files are in the data/ folder of the directory
    model_pkl = open(f"data/{model}.pkl", "rb")

    text_vector, classifier = pickle.load(model_pkl)

    original_text, tokenized_text = read_files(text_file, model)

    transformed_text = text_vector.transform(tokenized_text)

    pred_labels = classifier.predict(transformed_text)

    for i, sentence in enumerate(original_text):
        print(f"SENTENCE: {sentence}\t LABEL: {pred_labels[i]}")
 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_text_file = sys.argv[1]
    classifier = sys.argv[2]

    main(input_text_file, classifier)
