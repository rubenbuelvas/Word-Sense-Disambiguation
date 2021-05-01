import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB

# Feature extraction

CATEGORIES_STOPWORDS = ['./.', ',/,', '[', ']', '\n', "'", "''", '``']


def gc_extraction(n_words=2):
    data = []
    with open('interest.acl94.txt') as file:
        lines = file.readlines()
    separator = lines[1]
    lines.remove(separator)
    for line in lines:
        line = line.split(' ')
        line = list(filter(lambda x: x not in CATEGORIES_STOPWORDS, line))
        nulls = []
        for _ in range(n_words):
            nulls.append('/NULL')
        line = nulls + line + nulls
        target_word_found = False
        for i in range(len(line)):
            if line[i].find('interest_') == 0:
                category = line[i][9:10]
                line = line[i - n_words:i + n_words + 1]
                target_word_found = True
                break
        if target_word_found:
            line.pop(n_words)
            for i in range(len(line)):
                try:
                    line[i] = line[i].split('/')[1]
                except:
                    line[i] = 'NULL'
            line.insert(0, category)
            data.append(line)
    with open('gc_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


ENGLISH_STOPWORDS = []


def bow_extraction():
    pass


def load_dataset(filename, test_size=0.2):
    with open(filename) as file:
        file.readline()  # Ignore first line
        data = np.loadtxt(file)
    X = data[:, 1:]
    y = data[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test

# Naive Bayes

def train_naive_bayes(dataset):
    pass

# Decision Tree

# MultiLayer Perceptron


if __name__ == '__main__':
    gc_extraction()
