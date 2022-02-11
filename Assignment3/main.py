# Jonathan Chen, 20722167
# University of Waterloo

import sys
import os
import nltk
import nltk.tokenize as tok
from gensim.models import Word2Vec


def read_file(file_path):
    # Read data from files into a list
    with open(file_path + "/pos.txt") as file:
        pos_lines = file.readlines()
    with open(file_path + "/neg.txt") as file:
        neg_lines = file.readlines()
    
    data = pos_lines+neg_lines
    return data

def main(file_path):
    # Read in files
    data = read_file(file_path)

    # Tokenize the text
    nltk.download('punkt')

    print("Tokenizing the text")
    data = [tok.word_tokenize(line) for line in data]

    # Run the word2Vec model
    print("Training the Word2vec model")
    model = Word2Vec(data, min_count = 15, vector_size=200) 

    if not os.path.exists("data"):
        os.makedirs("data")

    # Save the model
    model.save('data/w2v.model')
    print(f"Model saved to {os.getcwd()}/data/w2v.model")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_folder_path = sys.argv[1]

    main(input_folder_path)
