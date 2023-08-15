# This is a single layer neural network for recognizing the language of given tet prompt. This is a solution to a given task which is included in the project as a txt file named task.txt


import os
import string
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

class LanguageClassifier:
    def __init__(self, train_data, test_data, learning_rate, num_epochs):
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.classes = dict()

    def make_unit_vector(self, vector):
        length = np.linalg.norm(vector)
        return vector / length

    def dot_product(self, v1, v2):
        return np.dot(v1, v2)

    def train(self):
        np.random.seed(42)
        for lang_idx, lang in enumerate(self.classes.keys()):
            weights = np.random.rand(len(self.train_data[0][0]))
            for epoch in range(self.num_epochs):
                for x, y_true in self.train_data:
                    y_pred = self.dot_product(x, weights)
                    if (y_true == lang) != (y_pred > 0):
                        weights = self.update_weights(weights, y_true, lang, x)
            self.classes[lang] = self.make_unit_vector(weights)


    def update_weights(self, weights, y_true, lang, x):
        x = np.array(x)
        if y_true == lang:
            return weights + self.learning_rate * x
        else:
            return weights - self.learning_rate * x


    def classify(self, text_vector):
        max_score = float("-inf")
        best_lang = None
        for lang, weights in self.classes.items():
            score = self.dot_product(text_vector, weights)
            if score > max_score:
                max_score = score
                best_lang = lang
        return best_lang

def get_text_vector(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = text.translate(str.maketrans("", "", string.punctuation)) # usuwam znaki interpunkcyjne
        counter = Counter(text)
        vector = [counter[char] for char in string.ascii_lowercase] # tworzę wektor częstotliwości występiwania liter
        return vector

def create_dataset(root_dir):
    dataset = []
    classes = {}
    for lang_dir in os.listdir(root_dir):
        lang_path = os.path.join(root_dir, lang_dir)
        if os.path.isdir(lang_path):
            lang_label = lang_dir.lower()
            classes[lang_label] = len(classes)
            for file_name in os.listdir(lang_path):
                file_path = os.path.join(lang_path, file_name)
                if os.path.isfile(file_path):
                    text_vector = get_text_vector(file_path)
                    dataset.append((text_vector, lang_label))
    return dataset, classes

def prepare_data(train_directory, test_directory):
    train_dataset, classes = create_dataset(train_directory)
    test_dataset, _ = create_dataset(test_directory)
    return train_dataset, test_dataset, classes

# User interface for entering short text for language classification
def classify_user_input(classifier):
    user_text = input("Enter short text for language classification: ")
    user_text_vector = Counter(user_text.lower().translate(str.maketrans("", "", string.punctuation)))
    user_text_vector = [user_text_vector[char] for char in string.ascii_lowercase]
    predicted_lang = classifier.classify(user_text_vector)
    print("Predicted language:", predicted_lang)

def main():
    train_directory = "train"
    test_directory = "test"
    learning_rate = 0.01
    num_epochs = 1000

    train_dataset, test_dataset, classes = prepare_data(train_directory, test_directory)

    classifier = LanguageClassifier(train_dataset, test_dataset, learning_rate, num_epochs)
    classifier.classes = classes
    classifier.train()

    # Test the classifier
    correct_predictions = 0
    total_predictions = 0
    for x, true_lang in test_dataset:
        predicted_lang = classifier.classify(x)
        if predicted_lang == true_lang:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)

    # Run the user interface
    classify_user_input(classifier)

if __name__ == "__main__":
    main()
