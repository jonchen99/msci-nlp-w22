# Assignment 2

Jonathan Chen
20722167

## Running the code:
To run the main file:
```python3 main.py [path to data folder]``` 

To run the inference file:
```python3 inference.py [path to txt file] [classifier to use]```

## Report

| Stopwords Removed | Text Features    | Accuracy (test set)|
| ----------------- | :---------------- | ------------------ |
| yes               | unigrams         | 80.71%             |
| yes               | bigrams          | 79.22&             |
| yes               | unigrams+bigrams | 82.46%             |
| no                | unigrams         | 80.89%             |
| no                | bigrams          | 82.49%             |
| no                | unigrams+bigrams | 83.29%             |
