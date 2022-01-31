# Jonathan Chen, 20722167
# University of Waterloo

import sys
import re 
import random
import os
import csv


def read_file(file_path, file_name):
    # Read data from files into a list
    with open(file_path + "/" + file_name) as file:
        lines = [line.strip() for line in file]

    # Append a 1 or 0 as the labels
    if file_name == 'pos.txt':
        lines = [[line, "1"] for line in lines]    
    else: 
        lines = [[line, "0"] for line in lines]    

    return lines

def write_file(file_path, data) :
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        writer.writerows(data)

def tokenize(data):
    # Remove any of these characters
    pattern = "[!\"#$%&()*+/:;<=>@[\\]^`{|}~\t\n]"

    # Stopwords from nltk stopwords
    stops = {'as', 'are', 'wasn', "wouldn't", 'those', 'and', 'himself', 'their', 've', 'at', 'off', 'by', "you'll", 'hers', 'what', 'be', 'few', 'aren', 'while', 'our', 'against', 'not', "haven't", 'to', 'an', 'haven', 'itself', 'into', 'with', 'through', "you're", 'theirs', 'themselves', 'but', "isn't", 'them', "hasn't", 'can', "shouldn't", "wasn't", 'wouldn', "mustn't", 'doing', 'll', 'if', 'any', 'she', 'once', 'down', 'm', 'having', 'shouldn', 'it', 'herself', 'am', 'during', 'between', 'about', 'where', 'who', 'from', 'we', 'my', 'shan', 'just', 'so', 'mustn', 'hadn', 'whom', "doesn't", 'hasn', 'after', "shan't", 'don', 'mightn', 'i', 'how', 'than', 's', 'again', 'of', 'for', "that'll", 'the', 'own', 'each', 'or', 'yours', 'he', 'there', 'most', 'ma', 'will', 'didn', 'here', 'that', 'a', 'why', 'y', "needn't", "you'd", 'ours', 'yourself', 'his', 'such', 'because', 'under', 'over', 'now', 'was', 'further', 'myself', 'very', 'couldn', 'has', 'both', 'did', 'in', 'on', 'doesn', 'had', 'you', 'above', 'him', 'been', 'no', 'before', 'this', 're', 'ourselves', "she's", 'other', 'only', "you've", 'its', 'below', 'then', 'ain', 'out', 'have', "won't", 'o', 'more', 'd', "weren't", 'does', "couldn't", "mightn't", 'me', 'yourselves', 'they', 'nor', 'is', 'weren', 'being', 'her', 'until', 'when', 't', "don't", "should've", 'same', 'too', "aren't", 'your', 'won', 'should', "hadn't", 'needn', 'which', 'these', "it's", 'do', 'were', 'isn', 'all', "didn't", 'up', 'some'}
    tokens = list()
    tokens_without_stopwords = list()
    labels = list()
    for line in data:
        # Remove characters
        contents = re.sub(pattern, "", line[0])

        # Tokenize by splitting on whitespace
        contents = contents.split()

        # Create list of tokens
        tokens.append(contents)
        tokens_without_stopwords.append([x for x in contents if x.lower() not in stops])

        # Create list of labels
        labels.append(line[1])

    return tokens, tokens_without_stopwords, labels

def main(file_path):
    # Read in files
    pos_text = read_file(file_path, 'pos.txt')
    neg_text = read_file(file_path, 'neg.txt')

    data = pos_text+neg_text

    # Randomize the data
    random.shuffle(data)
    tokens, tokens_without_stopwords, labels = tokenize(data)

    # Determine where to split the data
    percent80 = int(len(tokens)*0.8)
    percent90 = int(len(tokens)*0.9)

    # Split the data
    train = tokens[0:percent80]
    validation = tokens[percent80: percent90]
    test = tokens[percent90 :]

    if not os.path.exists("data"):
        os.makedirs("data")

    # Write to output files
    write_file("data/out.csv", tokens)
    write_file("data/train.csv", train)
    write_file("data/val.csv", validation)
    write_file("data/test.csv", test)

    train_no_stopwords = tokens_without_stopwords[0:percent80]
    validation_no_stopwords = tokens_without_stopwords[percent80 : percent90]
    test_no_stopwords = tokens_without_stopwords[percent90 :]

    write_file("data/out_ns.csv", tokens_without_stopwords)
    write_file("data/train_ns.csv", train_no_stopwords)
    write_file("data/val_ns.csv", validation_no_stopwords)
    write_file("data/test_ns.csv", test_no_stopwords)

    train_labels = labels[0:percent80]
    validation_labels = labels[percent80 : percent90]
    test_labels = labels[percent90 :]

    write_file("data/labels.csv", labels)
    write_file("data/train_labels.csv", train_labels)
    write_file("data/val_labels.csv", validation_labels)
    write_file("data/test_labels.csv", test_labels)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_folder_path = sys.argv[1]

    main(input_folder_path)
