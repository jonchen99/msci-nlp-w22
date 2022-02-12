# Jonathan Chen, 20722167
# University of Waterloo
# March 4, 2022

import re 
import warnings
import sys


def read_files(file_name, isLabel):
    # Read data from files into a list
    lines = []
    with open(file_name) as file:
        for line in file:
            string = re.sub("[\"\n]", "", line)
            if isLabel:
                lines.append(int(string))
            else:
                lines.append(string)
    
    return lines

def main(file_path):
    # Read in the input files
    train = read_files(file_path+"/train.csv", False)
    val = read_files(file_path + "/val.csv", False)
    test = read_files(file_path+"/test.csv", False)
    
    train_ns = read_files(file_path+"/train_ns.csv", False)
    val_ns = read_files(file_path + "/val_ns.csv", False)
    test_ns = read_files(file_path+"/test_ns.csv", False)

    train_labels = read_files(file_path+"/train_labels.csv", True)
    val_labels = read_files(file_path+"/val_labels.csv", True)
    test_labels = read_files(file_path+"/test_labels.csv", True)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("ERROR: Invalid number of inputs")
    #     exit(1)

    # input_folder_path = sys.argv[1]
    input_folder_path = "/Users/jonathanchen/Documents/School/4B/MSCI 598/Assignments/msci-nlp-w22/Assignment1/data"

    # warnings.filterwarnings("ignore")
    main(input_folder_path)
