# This is a simple perceptron. It was required to use linear regression and then quantify the results. This is a solutin to a given task attached to the folder as separate file task.txt

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


class Perceptron:
    def __init__(self, train_file, test_file, learning_rate, num_epochs):
        self.train_file = train_file
        self.test_file = test_file
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.accuracies = []
        self.train_accuracies = []
        self.classes = dict()

    def read_file(self, path):
        train = open(path)
        list_of_features = []
        labels = []
        for line in train:
            if not line.strip(): continue
            *features, label = line.split(',')
            features = list(map(float, features))
            list_of_features.append(features)
            labels.append(label.strip())
        return list_of_features, labels

    def training_step(self, ground_truth, y, X):
        assert len(self.weights) == len(X)
        for i, (w, x) in enumerate(zip(self.weights, X)):
            self.weights[i] = w + (ground_truth - y) * self.learning_rate * x # updating the weights
        if self.weights[0] == float('inf'):
            print('Weights became infinite, exiting.')
            exit(0)
        if self.weights[0] != self.weights[0]:
            print('Weights became NaN, exiting.')
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

    def dot_product(self, v1, v2):
        assert len(v1) == len(v2)
        sum = 0
        for i1, i2 in zip(v1, v2):
            sum += i1 * i2
        return sum

    def run_epoch(self, data, labels, train=False, print_predictions=False):
        accuracy = 0
        if train:
            self.learning_rate *= 0.98

        for observation, label in zip(data, labels):
            predicted, y, quant_y, X = self.predict(observation)
            ground_truth = self.classes[label]
            if train:
                self.training_step(ground_truth, quant_y, X) # passing quantified y into training step for weights actualization
                self.weights[::-1] = self.make_unit_vector(self.weights[::-1])
            elif print_predictions:
                print('true class: ', label, ', predicted: ', predicted)
            if ground_truth == quant_y:
                accuracy += 1
        return accuracy / len(data)

    def train(self):
        data, labels = self.read_file(self.train_file)
        self.weights = data[0] + [0.0]
        first_class = labels[0]
        self.classes[first_class] = 0
        for X, label in zip(data, labels):
            if label != first_class:
                second_class = label
                break
        self.classes[second_class] = 1
        for i in range(self.num_epochs):
            train_accuracy = self.run_epoch(data, labels, train=True) # run_epoch
            self.train_accuracies.append(train_accuracy)
            pairs = list(zip(data, labels))
            random.Random(42).shuffle(pairs)
            data, labels = zip(*pairs)
            self.accuracies.append(self.test())

    def test(self, print_predictions=False):
        data, labels = self.read_file(self.test_file)
        return self.run_epoch(data, labels, train=False, print_predictions=print_predictions)

    def predict(self, observation):
        X = observation + [1.0]
        y = self.dot_product(self.weights, X)
        if y > 0.5:
            quant_y = 1
        else:
            quant_y = 0
        predicted = list(self.classes.keys())[list(self.classes.values()).index(quant_y)]
        return predicted, y, quant_y, X

    def accuracy_plot(self):
        plt.figure(figsize=(10, 8))
        plt.title(f'Accuracies after each epoch for learning rate {str(self.learning_rate)}.')
        plt.plot(list(range(1, self.num_epochs + 1)), self.accuracies, label='Test set')
        plt.plot(list(range(1, self.num_epochs + 1)), self.train_accuracies, label='Training set')
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        plt.show()
        # for epoch_num, accuracy in enumerate(self.accuracies):
        #     print(f'Accuracy for {epoch_num} epoch: {accuracy}')

    def sklearn_logistic_regression(self):
        """ Train and measure sklearn LogisticRegression for comparison. """
        train_features, train_labels = self.read_file(self.train_file)
        test_features, test_labels = self.read_file(self.test_file)

        logreg = LogisticRegression(random_state=42)  # , max_iter=10000
        logreg.fit(train_features, train_labels)
        predictions = logreg.predict(test_features)

        # calculate and draw confusion matrix heatmap
        cnf_matrix = metrics.confusion_matrix(test_labels, predictions)
        class_names = [0, 1]
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.title('Confusion matrix (sklearn logistic regression results)', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

        print(f'End accuracy for sklearn logistic regression: {metrics.accuracy_score(predictions, test_labels)}')


def main():
    learning_rate = 0.002
    epochs_num = 100
    experiment_example2_perceptron(epochs_num, learning_rate)
    learning_rate = 0.005
    epochs_num = 200
    p = experiment_iris_perceptron(epochs_num, learning_rate)
    # UI
    answer = True
    while answer:
        print('Wybierz: ')
        print('\t0, jeśli chcesz wyjść z programu')
        print('\t1, jeśli chcesz podać swoją obserwację')
        print('\t2. jeśli chcesz zmenić stałą uczenia')
        answer = int(input('Wybierz (wpisz tylko jeden znak): '))
        match answer:
            case 1:
                observation = input('Podaj atrybuty obserwacji oddzielone przecinkami: ')
                observation = list(map(float, observation.split(',')))
                print('Predicted value for your observation is: ', p.predict(observation)[0])
            case 2:
                learning_rate = float(input('Podaj nową stałą uczenia: '))
                experiment_example2_perceptron(epochs_num, learning_rate)
                p = experiment_iris_perceptron(epochs_num, learning_rate)


def experiment_iris_perceptron(epochs_num, learning_rate):
    print('\nPerceotron performance for data from iris_perceptron:\n')
    p = Perceptron('data\\iris_perceptron\\training.txt', 'data\\iris_perceptron\\test.txt', learning_rate, epochs_num)
    p.train()
    p.accuracy_plot()
    p.sklearn_logistic_regression()
    return p


def experiment_example2_perceptron(epochs_num, learning_rate):
    print('\nPerceotron performance for data from example2:\n')
    p = Perceptron('data\\example2\\train.txt', 'data\\example2\\test.txt', learning_rate, epochs_num)
    p.train()
    p.accuracy_plot()
    draw_dataset_and_decision_line(p)
    p.sklearn_logistic_regression()


def draw_dataset_and_decision_line(p):
    rows = []
    for feat, lbl in zip(*p.read_file(p.train_file)):
        assert len(feat) == 2
        rows.append(dict(x=feat[0], y=feat[1], lbl=lbl, split='train'))
    for feat, lbl in zip(*p.read_file(p.test_file)):
        assert len(feat) == 2
        rows.append(dict(x=feat[0], y=feat[1], lbl=lbl, split='test'))
    df = pd.DataFrame(rows)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df, x='x', y='y', hue='lbl', style='split')
    A, B, C = p.weights
    xs = np.linspace(df.x.min(), df.x.max(), 100)
    ys = (0.5 - A * xs - C) / B
    plt.plot(xs, ys)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
