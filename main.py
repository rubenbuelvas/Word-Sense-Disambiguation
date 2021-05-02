import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

count_vectorizer = CountVectorizer()
tfid_vectorizer = TfidfVectorizer()
CATEGORIES_STOPWORDS = ['./.', ',/,', '[', ']', '(/(', ')/)', '\n', "'/'", "''/''", '``/``', ':/:', ';/:', '--/:', '{/(', '}/)']
ENGLISH_STOPWORDS = []
test_size = 0.5
gc_dataset_filename = 'gc_dataset.csv'
bow_dataset_filename = 'bow_dataset.csv'

# Feature extraction


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
    X = count_vectorizer.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test


# Naive Bayes

def train_naive_bayes(dataset_filename):
    X_train, X_test, y_train, y_test = load_dataset(dataset_filename)
    nb = CategoricalNB()
    nb.fit(X_test, y_test)
    print(X_train)
    print(str(np.shape(X_train)) + ' ' + str(np.shape(X_test)) + ' ' + str(np.shape(y_train)) + ' ' + str(np.shape(y_test)))
    y_pred = nb.predict(X_train)
    print(y_pred)
    print('Accuracy: ' + str(accuracy_score(y_train, y_pred)))


# Decision Tree

# MultiLayer Perceptron


if __name__ == '__main__':
    gc_extraction()
    train_naive_bayes(gc_dataset_filename)
