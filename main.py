import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier

count_vectorizer = CountVectorizer()
tfid_vectorizer = TfidfVectorizer()
CATEGORIES_STOPWORDS = ['./.', ',/,', '[', ']', '(/(', ')/)', '\n', "'/'", "''/''", '``/``', ':/:', ';/:', '--/:',
                        '{/(', '}/)']
ENGLISH_STOPWORDS = []
test_size = 0.3
n_words = 2
gc_dataset_filename = 'gc_dataset.csv'
bow_dataset_filename = 'bow_dataset.csv'


# Feature extraction


def gc_extraction():
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
            nulls.append('/VOID')
        line = nulls + line + nulls
        target_word_found = False
        for i in range(len(line)):
            if line[i].find('interest_') == 0:
                category = 'C' + line[i][9:10]
                line = line[i - n_words:i + n_words + 1]
                target_word_found = True
                break
        if target_word_found:
            line.pop(n_words)
            for i in range(len(line)):
                try:
                    line[i] = line[i].split('/')[1]
                except:
                    line[i] = 'VOID'
            line = [' '.join(line)]
            line.insert(0, category)
            data.append(line)
    with open('gc_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def bow_extraction():
    pass


def load_dataset(filename):
    data = pd.read_csv(filename, header=None)
    y = data[0].values
    X = data[1]
    X = tfid_vectorizer.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test


# Naive Bayes

def naive_bayes(dataset_filename):
    X_train, X_test, y_train, y_test = load_dataset(dataset_filename)
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print('NB Accuracy: ' + str(accuracy_score(y_test, y_pred)))


# Decision Tree

def decision_tree(dataset_filename):
    X_train, X_test, y_train, y_test = load_dataset(dataset_filename)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print('DT Accuracy: ' + str(accuracy_score(y_test, y_pred)))


# MultiLayer Perceptron

def multilayer_perceptron(dataset_filename):
    X_train, X_test, y_train, y_test = load_dataset(dataset_filename)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 3), random_state=1, max_iter=2000)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print('MLP Accuracy: ' + str(accuracy_score(y_test, y_pred)))


if __name__ == '__main__':
    gc_extraction()
    naive_bayes(gc_dataset_filename)
    decision_tree(gc_dataset_filename)
    multilayer_perceptron(gc_dataset_filename)
