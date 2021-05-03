import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import cross_validate

# Constants

lancaster_stemmer = LancasterStemmer()
CATEGORIES_STOPWORDS = ['/.', '/,', '[', ']', '/(', '/)', '\n', "/'", "/''", '/``', '/:']
CATEGORIES_STOPWORDS_EXTRA = ['/IN', '/DT', '/CC']
PUNCTUATION_STOPWORDS = ['.', ',', '[', ']', '(', ')', '\n', "'", "''", '``', ':', ';', '{', '}']
ENGLISH_STOPWORDS = []
test_size = 0.3
n_words = 3

TRAIN_CONFIG = {
    'stopwords': 'all',  # ['all', 'punctuation', 'none']
    'gc': {
        'filename': 'gc_dataset.csv',
    },
    'bow': {
        'filename': 'bow_dataset.csv',
    },
    'tfidf_auto': {

    },
    'custom': {

    }
}

# Feature extraction

def generate_datasets():
    gc_extraction()
    bow_extraction()


# Grammatical classification

def gc_extraction(stopwords=True, extra_stopwords=True):
    data = []
    with open('interest.acl94.txt') as file:
        lines = file.readlines()
    separator = lines[1]
    lines.remove(separator)
    for line in lines:
        tmp = []
        line = line.split(' ')
        try:
            line.remove('======================================')
        except:
            pass
        if stopwords:
            for i in range(len(line)):
                stopword_found = False
                for stopword in CATEGORIES_STOPWORDS:
                    if line[i].find(stopword) != -1:
                        stopword_found = True
                        break
                if not stopword_found and extra_stopwords:
                    for stopword in CATEGORIES_STOPWORDS_EXTRA:
                        if line[i].find(stopword) != -1:
                            stopword_found = True
                            break
                if not stopword_found:
                    tmp.append(line[i])
            line = tmp
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
            elif line[i].find('interests_') == 0:
                category = 'C' + line[i][10:11]
                line = line[i - n_words:i + n_words + 1]
                line.pop(n_words)
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


# Bag of words

def bow_extraction(punctuation_stopwords=True, english_stopwords=True):
    data = []
    with open('interest-original.txt') as file:
        lines = file.readlines()
    separator = lines[1]
    lines.remove(separator)
    for line in lines:
        line = word_tokenize(line)
        try:
            line.remove('======================================')
        except:
            pass
        for i in range(len(line)):
            line[i] = lancaster_stemmer.stem(line[i])
        if punctuation_stopwords:
            line = list(filter(lambda x: x not in PUNCTUATION_STOPWORDS, line))
        nulls = []
        for _ in range(n_words):
            nulls.append('VOID')
        line = nulls + line + nulls
        target_word_found = False
        for i in range(len(line)):
            if line[i].find('interest_') == 0:
                category = 'C' + line[i][9:10]
                line = line[i - n_words:i + n_words + 1]
                line.pop(n_words)
                target_word_found = True
                break
            elif line[i].find('interests_') == 0:
                category = 'C' + line[i][10:11]
                line = line[i - n_words:i + n_words + 1]
                line.pop(n_words)
                target_word_found = True
                break
            elif not line[i].isalpha():
                line[i] = 'NUM'
        if target_word_found:
            line.insert(0, category)
            data.append(line)
    with open('bow_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


# TF-IDF feature extraction

def tfidf_auto_extraction(english_stopwords=True):
    y = []
    with open('interest-original.txt') as file:
        lines = file.readlines()
    separator = lines[1]
    lines.remove(separator)
    for line in lines:
        line = line.replace('======================================', '')
        tk_line = word_tokenize(line)
        target_word_found = False
        for i in range(len(tk_line)):
            if tk_line[i].find('interest_') == 0:
                y.append('C' + tk_line[i][9:10])
                line = line.replace(tk_line[i], 'interest')
                break
            elif tk_line[i].find('interests_') == 0:
                y.append('C' + tk_line[i][10:11])
                line = line.replace(tk_line[i], 'interests')
                break
    with open('tfidf_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)


# TF-IDF + GC

def tfidf_gc_extraction():
    pass


# Test routine

def run_all_models(dataset):
    stopwords = TRAIN_CONFIG.get('stopwords')
    config = TRAIN_CONFIG.get(dataset)
    data = pd.read_csv(config.get('filename'), header=None)
    y = data[0].values
    X = data[1]
    if dataset == 'tfidf_auto':
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    acc, f1 = naive_bayes(X_train, X_test, y_train, y_test)
    print('NB Accuracy: ' + str('{:.4f}'.format(acc)))
    acc, f1 = decision_tree(X_train, X_test, y_train, y_test)
    print('DT Accuracy: ' + str('{:.4f}'.format(acc)))
    acc, f1 = multilayer_perceptron(X_train, X_test, y_train, y_test)
    print('MLP Accuracy: ' + str('{:.4f}'.format(acc)))


def find_optimal_dt(dataset):
    pass


def find_optimal_mlp(dataset):
    pass


# Naive Bayes

def naive_bayes(X_train, X_test, y_train, y_test):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')


# Decision Tree

def decision_tree(X_train, X_test, y_train, y_test, depth=-1):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')


# MultiLayer Perceptron

def multilayer_perceptron(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100, 10)):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')


# Main routine

if __name__ == '__main__':
    generate_datasets()
    print('GC')
    run_all_models('gc')
    print('BOW')
    run_all_models('bow')
    tfidf_auto_extraction()
