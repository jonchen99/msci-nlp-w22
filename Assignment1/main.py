import sys
import re 
import random
from nltk.corpus import stopwords
import os
import numpy as np

def read_file(file_path, file_name):
    with open(file_path + file_name) as file:
        lines = [line.strip() for line in file]

    if file_name == 'pos.txt':
        lines = [[line, 1] for line in lines]    
    else: 
        lines = [[line, 0] for line in lines]    

    return lines

def tokenize(data):
    pattern = "[!\"#$%&()*+/:;<=>@[\\]^`{|}~\t\n]"
    stops = set(stopwords.words('english'))
    tokens = list()
    tokens_without_stopwords = list()
    labels = list()
    for line in data:
        contents = re.sub(pattern, "", line[0])
        contents = contents.split()
        tokens.append(contents)

        tokens_without_stopwords.append([x for x in contents if x.lower() not in stops])

        labels.append(line[1])

    return tokens, tokens_without_stopwords, labels

def main(file_path):
    pos_text = read_file(file_path, 'pos.txt')
    neg_text = read_file(file_path, 'neg.txt')

    data = pos_text+neg_text

    random.shuffle(data)
    tokens, tokens_without_stopwords, labels = tokenize(data)

    percent80 = int(len(tokens)*0.8)
    percent90 = int(len(tokens)*0.9)

    train = tokens[0:percent80]
    validation = tokens[percent80: percent90]
    test = tokens[percent90 :]

    if not os.path.exists("data"):
        os.makedirs("data")

    np.savetxt("data/out.csv", tokens, delimiter=',', fmt = '%s')
    np.savetxt("data/train.csv", train, delimiter=',', fmt='%s')
    np.savetxt("data/val.csv", validation, delimiter=',', fmt='%s')
    np.savetxt("data/test.csv", test, delimiter=',', fmt='%s')

    train_no_stopwords = tokens_without_stopwords[0:percent80]
    validation_no_stopwords = tokens_without_stopwords[percent80 : percent90]
    test_no_stopwords = tokens_without_stopwords[percent90 :]

    np.savetxt("data/out_ns.csv", tokens_without_stopwords, delimiter=',', fmt = '%s')
    np.savetxt("data/train_ns.csv", train_no_stopwords, delimiter=',', fmt='%s')
    np.savetxt("data/val_ns.csv", validation_no_stopwords, delimiter=',', fmt='%s')
    np.savetxt("data/test_ns.csv", test_no_stopwords, delimiter=',', fmt='%s')

    train_labels = labels[0:percent80]
    validation_labels = labels[percent80 : percent90]
    test_labels = labels[percent90 :]

    np.savetxt("data/labels.csv", labels, delimiter=',', fmt = '%s')
    np.savetxt("data/train_labels.csv", train_labels, delimiter=',', fmt='%s')
    np.savetxt("data/val_labels.csv", validation_labels, delimiter=',', fmt='%s')
    np.savetxt("data/test_labels.csv", test_labels, delimiter=',', fmt='%s')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_folder_path = sys.argv[1]

    main(input_folder_path)
