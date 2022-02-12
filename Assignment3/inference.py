# Jonathan Chen, 20722167
# University of Waterloo
# February 18, 2022

import re
import sys
from gensim.models import Word2Vec

def read_file(file_name):
    # Read data from files into a list
    words = []

    with open(file_name) as file:
        for line in file:
            words.append(re.sub("[\n]", "", line))

    return words

def main(text_file):
    # Assumes the word2vec model is in the data/ folder of the directory
    model = Word2Vec.load(f"data/w2v.model")

    # Read in the words
    words = read_file(text_file)

    for word in words:
        # Print the most similar words
        try:
            similar_words = model.wv.most_similar(word, topn=20)
            print(f"Top 20 most similar words to {word} are: ")
            for similar_word in similar_words:
                print(f'{similar_word[0]}: {similar_word[1]}')
        except:
            print(f"Unable to find similar words for {word}")
        print()
 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_text_file = sys.argv[1]

    main(input_text_file)
