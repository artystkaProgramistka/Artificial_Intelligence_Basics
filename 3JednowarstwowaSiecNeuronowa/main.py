import os
import string
import matplotlib.pyplot as plt
from collections import Counter
from random import shuffle
from copy import deepcopy

class Perceptron:
    def __init__(self, train_data, test_data, learning_rate, num_epochs):
        self.test_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = 1
        self.accuracies = []
        self.train_accuracies = []
        self.classes = dict()

    def delta(self, d, y, X):
        for i, (w, x) in enumerate(zip(self.weights, X)):
            self.weights[i] = w + (d - y) * self.learning_rate * x
        if self.weights[0] == float('inf'):
            print('Weights became infinite, exiting.')
            exit(0)

    def make_unit_vector(self, vector):
        length = 0
        for elem in vector:
            length += elem ** 2
        length = length ** (1 / 2)
        for i, elem in enumerate(vector):
            elem /= length
            vector[i] = elem
        return vector

    def train(self):
        self.weights = self.test_data[0][:-1] + [self.threshold, ]
        first_class = self.test_data[0][-1] # eng
        self.classes[first_class] = 0
        for X in self.test_data:
            if X[-1] != first_class:
                second_class = X[-1] # not_eng
                break
        self.classes[second_class] = 1
        for i in range(self.num_epochs):
            # weights -> unit vector
            # w = np.array(self.weights)
            # self.weights = w / np.linalg.norm(w)
            self.weights = self.make_unit_vector(self.weights)
            train_accuracy = 0
            for observation in self.test_data:
                X = observation[:-1] + [-1, ]
                y = self.dot_product(self.weights, X)
                if y >= self.threshold:
                    quant_y = 1
                else:
                    quant_y = 0
                d = self.classes[observation[-1]]
                self.delta(d, y, X)
                self.threshold = self.weights[-1]
                if d == quant_y:
                    train_accuracy += 1
            self.train_accuracies.append(train_accuracy / len(self.test_data))
            shuffle(self.test_data)
            self.accuracies.append(self.test())

    def dot_product(self, v1, v2):
        sum = 0
        for i1, i2 in zip(v1, v2):
            sum += i1 * i2
        return sum

    def test(self, print_predictions=False):
        accuracy = 0
        for observation in self.test_data:
            X = observation[:-1] + [-1, ]
            y = self.dot_product(self.weights, X)
            if y > self.threshold:
                y = 1
            else:
                y = 0
            predicted = list(self.classes.keys())[list(self.classes.values()).index(y)]
            if print_predictions:
                print('true class: ', observation[-1], ', predicted: ', predicted)
            if predicted == observation[-1]:
                accuracy += 1
        return accuracy / len(self.test_data)

    def give_prediction(self, observation):
        X = observation
        y = self.dot_product(self.weights, X)
        if y > self.threshold:
            y_quant = 1
        else:
            y_quant = 0
        predicted = list(self.classes.keys())[list(self.classes.values()).index(y_quant)]
        return predicted, y

    def accuracy_plot(self):
        plt.figure(figsize=(20, 10))
        plt.title(f'Accuracies after each epoch for learning rate {str(self.learning_rate)}.')
        plt.plot(list(range(1, self.num_epochs + 1)), self.accuracies, label='Test set')
        plt.plot(list(range(1, self.num_epochs + 1)), self.train_accuracies, label='Training set')
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        plt.show()
        for epoch_num, accuracy in enumerate(self.accuracies):
            print(f'Accuracy for {epoch_num} epoch: {accuracy}')
def get_text_vector(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = text.translate(str.maketrans("", "", string.punctuation)) # usuwamy znaki interpunkcyjne
        counter = Counter(text)
        vector = [counter[char] for char in string.ascii_lowercase] # tworzymy wektor częstotliwości wystąpień literek
        return vector

def create_dataset(root_dir):
    dataset = []
    num_langs = 0
    for lang_dir in os.listdir(root_dir):
        num_langs += 1
        lang_path = os.path.join(root_dir, lang_dir)
        if os.path.isdir(lang_path):
            lang_label = lang_dir.lower()
            for file_name in os.listdir(lang_path):
                file_path = os.path.join(lang_path, file_name)
                if os.path.isfile(file_path):
                    text_vector = get_text_vector(file_path)
                    dataset.append(text_vector + [lang_label,])
    return dataset, num_langs


def prepare_data(root_directory):
    dataset, num_langs = create_dataset("train")
    print(dataset)
    lang = dataset[0][-1]
    langs = [lang, ]
    datasets = [deepcopy(dataset), ]
    for data in datasets[0]:
        if data[-1] != lang:
            data[-1] = 'not_' + lang
    for i in range(1, num_langs):
        datasets.append(deepcopy(dataset))
        for data in datasets[i]:
            if data[-1] in langs or data[-1][:4] == 'not_':
                continue
            else:
                lang = data[-1]
                langs.append(lang)
        for data in datasets[i]:
            if data[-1] != langs[i]:
                data[-1] = 'not_' + langs[i]
    return datasets, dataset

def main():
    # preparing data for the perceptron layer training:
    train_datasets, original_train_dataset = prepare_data("train")
    # preparing data for the perceptron network testing:
    test_datasets, original_train_dataset = prepare_data("test")
    label1 = original_train_dataset[0][-1]
    # creating the layer of perceptrons:
    perceptrons = []
    for i, (train_data, test_data) in enumerate(zip(train_datasets, test_datasets)):
        perceptron = Perceptron(train_data, test_data, learning_rate=0.003, num_epochs=1_000)
        perceptrons.append(perceptron)
        perceptron.train()

    result = 'not_'
    confidence = 0
    for i, (perceptron, test_data) in enumerate(zip(perceptrons, test_datasets)):
        print(f'\nPERCEPTRON {i}:\n')
        data = test_data[0][:-1]
        label = test_data[0][-1]
        predicted, y = perceptron.give_prediction(data)
        print(f'predicted: {predicted}, actual: {label}')
        if y > confidence:
            result = predicted if not predicted.startswith('not_') else predicted[4:]
            confidence = y
    print(f'\nPrediction of the layer of perceptrons is {result}, with the confidence of {confidence}, actual: {label1}')




if __name__ == '__main__':
    main()


