import pickle
import re

def read_files(file_name):
    # Read data from files into a list
    lines = []
    with open(file_name) as file:
        for line in file:
            string = re.sub("[\"\n]", "", line)
            lines.append(string)
    
    return lines

def main(text_file, model):
    # TODO: UNHARDCODE THE DATA FOLDER PATH
    data_folder =  "/Users/jonathanchen/Documents/School/4B/MSCI 598/Assignments/msci-nlp-w22/Assignment2/data"
    # model_pkl = open(f"{data_folder}/{model}.pkl", "rb")

    model_pkl = open(f"data/{model}.pkl", "rb")

    text_vector, classifier = pickle.load(model_pkl)

    text = read_files(text_file)
    transformed_text = text_vector.transform(text)
    pred_labels = classifier.predict(transformed_text)
    print(pred_labels)

if __name__ == "__main__":
    # TODO: UNCOMMENT THIS
    # if len(sys.argv) != 3:
    #     print("ERROR: Invalid number of inputs")
    #     exit(1)

    # input_text_file = sys.argv[1]
    # classifier = sys.argv[2]

    input_text_file =  "/Users/jonathanchen/Documents/School/4B/MSCI 598/Assignments/Assignment2/sample_review.txt"
    model = "mnb_uni_bi"
    

    main(input_text_file, model)
