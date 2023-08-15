# This Python script is a handcrafted implementation of the K-Nearest Neighbors (KNN) algorithm, a popular machine learning model used for both regression and classification tasks. The implementation is not reliant on any machine learning libraries and solely uses basic Python functionality, and the matplotlib library for visualization.

import matplotlib.pyplot as plt

def read_file(path):
    train = open(path)
    lines = []
    for line in train:
        *features, label = line.split(',')
        features = list(map(float, features))
        lines.append(features + [label.strip()])
    return lines


def compute_neighbours(k, file, new_observation):
    train = read_file(file)
    pairs = []
    for observation in train:
        distance = 0
        for i in range(len(observation) - 1):
            distance += (observation[i] - new_observation[i]) ** 2
        pairs.append((distance, observation[len(observation) - 1]))
    pairs.sort()
    k_nearest = []

    counts = dict()
    for i in range(k):
        k_nearest.append(pairs[i])
        if pairs[i][1] in counts:
            counts[pairs[i][1]] += 1
        else:
            counts[pairs[i][1]] = 1

    return max(counts.items(), key=lambda kv: kv[1])[0]


def KNN(k, train_file, test_file):
    test = read_file(test_file)
    count = 0
    for observation in test:
        result = compute_neighbours(k, train_file, observation)
        print(', '.join(map(str, observation)), '\tpredicted: ', result)
        if result == observation[-1]:
            count += 1

    accuracy = count / len(test) * 100
    return accuracy


def optimize(smallest_k, largest_k, train_file, test_file):
    accuracies = dict()
    best_k = smallest_k
    best_accuracy = KNN(smallest_k, train_file, test_file)
    accuracies[smallest_k] = best_accuracy
    for k in range(smallest_k + 1, largest_k):
        current_accuracy = KNN(k, train_file, test_file)
        accuracies[k] = current_accuracy
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_k = k
    # draw a plot
    plt.figure(figsize=(20, 10))
    plt.title('Accuracy of KNN for various k')
    plt.plot(list(accuracies.keys()), list(accuracies.values()))
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    return best_k


def UI():
    train_file = 'train.txt'
    test_file = 'test.txt'
    k = optimize(1, 20, train_file, test_file)
    print('wyniki KNN dla zbioru testowego gdy k= ', k)
    print('\nDokładność: ', KNN(k, train_file, test_file), ' %')
    flag = True
    while flag:
        flag = True
        print('\nWybierz: ')
        print(
            '1, jeśli chcesz wpisać własną obserwację i uzyskać przewidywaną klasę na podstawie zbioru treningowego')
        print('2, jeśli chcesz zmienić k')
        print('3, jeśli chcesz zakończyć program')
        answer = int(input('Wpisz tylko jeden znak: '))
        if answer == 1:
            observation = input('Wpisz własną obserwację. Atrybuty oddziel przecinkami: ')
            observation = observation.split(',')
            observation = list(map(float, observation))
            observation += ', null'
            print('Przewidywana klasa dla twojej obserwacji to: ',
                  compute_neighbours(k, train_file, observation))
        elif answer == 2:
            k = int(input('Podaj nowe k: '))
            print('wyniki KNN dla zbioru testowego gdy k= ', k)
            print('\nDokładność: ', KNN(k, train_file, test_file), ' %')
        elif answer == 3:
            flag = False
        else:
            print('Wpisałeś nieodpowiedni znak, spróbuj jeszcze raz.')


# Opcjonalnie: Utwórz wykres dokładności od k

if __name__ == '__main__':
    UI()
