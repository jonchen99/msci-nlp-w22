# Jonathan Chen, 20722167
# University of Waterloo
# February 11, 2022

import re 
import pickle
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import numpy as np
import warnings

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

def create_text_vectors(feature, train, val, test):
    
    if feature == "unigrams":
        count_vect = CountVectorizer(ngram_range=(1,1))
    elif feature == "bigrams":
        count_vect = CountVectorizer(ngram_range=(2,2))
    elif feature == "unigrams+bigrams":
        count_vect = CountVectorizer(ngram_range=(1,2))
    else:
        print("Unable to create text vectors")
        exit(1)
    
    X_train = count_vect.fit_transform(train)
    X_val = count_vect.transform(val)
    X_test = count_vect.transform(test)

    return X_train, X_val, X_test

def train_model(X_train, X_val, train_labels, val_labels):
    # INITIAL PARAMETER SEARCH
    # alpha_range = [0, 0.5, 1, 2, 5, 10, 25, 50, 100]

    # # SECOND PARAMETER SEARCH
    alpha_range = np.linspace(0, 1.5, 16)

    best_alpha = 0
    best_accuracy = 0

    for alpha in alpha_range:
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train, train_labels)
        Y_pred = clf.predict(X_val)
        if (accuracy_score(val_labels, Y_pred) > best_accuracy):
            best_accuracy = accuracy_score(val_labels, Y_pred)
            best_alpha = alpha
        
        # print(alpha, " = ", accuracy_score(val_labels, Y_pred))
    
    best_model = MultinomialNB(alpha=best_alpha)
    best_model.fit(X_train, train_labels)
    return best_model

def test_model(clf, X_test, test_labels, model):
    Y_pred = clf.predict(X_test)

    print(model + ": Accuracy Score is " + str(accuracy_score(test_labels, Y_pred)))

def run_models(train, val, test, train_ns, val_ns, test_ns, train_labels, val_labels, test_labels):
    models = ['unigrams stopwords', 'bigrams stopwords', 'unigrams+bigrams stopwords',
                'unigrams no stopwords', 'bigrams no stopwords', 'unigrams+bigrams no stopwords']
    
    output_filenames = ['mnb_uni.pkl', 'mnb_bi.pkl', 'mnb_uni_bi.pkl', 'mnb_uni_ns.pkl', 'mnb_bi_ns.pkl', 'mnb_uni_bi_ns.pkl']

    if not os.path.exists("data"):
        os.makedirs("data")

    for i, model in enumerate(models):
        if model.split(" ")[1] == 'stopwords':
            X_train, X_val, X_test = create_text_vectors(model.split(" ")[0], train, val, test)
            best_model = train_model(X_train, X_val, train_labels, val_labels)
            
            test_model(best_model, X_test, test_labels, model)
        else:
            X_train, X_val, X_test = create_text_vectors(model.split(" ")[0], train_ns, val_ns, test_ns)
            best_model = train_model(X_train, X_val, train_labels, val_labels)
            
            test_model(best_model, X_test, test_labels, model)
        
        model_pkl = open("data/" + output_filenames[i], "wb")
        pickle.dump(best_model, model_pkl)
        model_pkl.close()

def main(file_path):
    train = read_files(file_path+"/train.csv", False)
    val = read_files(file_path + "/val.csv", False)
    test = read_files(file_path+"/test.csv", False)
    
    train_ns = read_files(file_path+"/train_ns.csv", False)
    val_ns = read_files(file_path + "/val_ns.csv", False)
    test_ns = read_files(file_path+"/test_ns.csv", False)

    train_labels = read_files(file_path+"/train_labels.csv", True)
    val_labels = read_files(file_path+"/val_labels.csv", True)
    test_labels = read_files(file_path+"/test_labels.csv", True)

    run_models(train, val, test, train_ns, val_ns, test_ns, train_labels, val_labels, test_labels)
   

    # nb = naive_bayes.MultinomialNB()
    # clf = GridSearchCV(nb, param_grid= {'alpha': [1,2,3]})
    # clf.fit(X_train, train_labels)
    # y_pred = clf.predict(X_val)

if __name__ == "__main__":
    # TODO: UNCOMMENT THIS
    # if len(sys.argv) != 2:
    #     print("ERROR: Invalid number of inputs")
    #     exit(1)

    # input_folder_path = sys.argv[1]
    input_folder_path =  "/Users/jonathanchen/Documents/School/4B/MSCI 598/Assignments/msci-nlp-w22/Assignment1/data"
    
    # TODO REMOVE THIS
    # model_pkl = open("data/mnb_uni_ns.pkl", "rb")
    # print(pickle.load(model_pkl))

    warnings.filterwarnings("ignore")
    main(input_folder_path)
