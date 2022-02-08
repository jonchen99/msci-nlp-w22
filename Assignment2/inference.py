# Jonathan Chen, 20722167
# University of Waterloo
# February 11, 2022

import pickle
import re
import sys

def read_files(file_name):
    # Read data from files into a list
    lines = []
    with open(file_name) as file:
        for line in file:
            string = re.sub("[\"\n]", "", line)
            lines.append(string)
    
    return lines

def main(text_file, model):
    # Assumes the pickle files are in the data/ folder of the directory
    model_pkl = open(f"data/{model}.pkl", "rb")

    text_vector, classifier = pickle.load(model_pkl)

    text = read_files(text_file)
    transformed_text = text_vector.transform(text)
    pred_labels = classifier.predict(transformed_text)
    print(pred_labels)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_text_file = sys.argv[1]
    classifier = sys.argv[2]

    main(input_text_file, classifier)
