# Feature extraction

CATEGORIES_STOPWORDS = ['./.', ',/,', '[', ']', '\n']


def categories_extraction(n_words=2):
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
        for i in range(len(line)):
            if line[i].find('interest_') == 0:
                category = line[i][9:10]
                line = line[i-n_words:i+n_words+1]
                break
        line.pop(n_words)
        for i in range(len(line)):
            line[i] = line[i].split('/')[1]
        line.append(category)
        print(line)
        break


# Naive Bayes Training

# Naive Bayes Test

# Decision Tree Training

# Decision Tree Test

# MultiLayer Perceptron Training

# MultiLayer Perceptron Test


if __name__ == '__main__':
    categories_extraction()
