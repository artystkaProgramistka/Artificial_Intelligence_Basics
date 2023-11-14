import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class CategoricalNaiveBayes:
    def __init__(self, laplace_smoothing=1):
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.feature_prob = {}
        self.class_prob = {}

        # Initialize probabilities
        for c in self.classes:
            self.feature_prob[c] = {}
            for feature in range(X.shape[1]):
                self.feature_prob[c][feature] = {}

        # Calculate class probabilities
        class_counts = np.bincount(y)
        self.class_prob = class_counts / len(y)

        # Calculate feature probabilities for each class
        for c in self.classes:
            features_in_class = X[y == c]
            total = features_in_class.shape[0] + self.laplace_smoothing * len(np.unique(X[:, feature]))
            for feature in range(X.shape[1]):
                counts = np.bincount(features_in_class[:, feature], minlength=total)
                probabilities = (counts + self.laplace_smoothing) / total
                self.feature_prob[c][feature] = probabilities

    def predict(self, X):
        predictions = []
        for instance in X:
            class_probabilities = self.class_prob.copy()
            for c in self.classes:
                for feature in range(X.shape[1]):
                    relative_feature_value = instance[feature]
                    class_probabilities[c] *= self.feature_prob[c][feature][relative_feature_value]
            predictions.append(np.argmax(class_probabilities))
        return predictions

def main():
    # Load the data
    file_path = 'car_evaluation.data'
    cars_data = pd.read_csv(file_path, header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptance'])

    # Convert categorical data to numerical data using LabelEncoder
    encoders = {}
    for column in cars_data.columns:
        encoder = LabelEncoder()
        cars_data[column] = encoder.fit_transform(cars_data[column])
        encoders[column] = encoder  # Store the encoder

    # Define features (X) and target (y)
    X = cars_data.drop('acceptance', axis=1).values
    y = cars_data['acceptance'].values

    accuracies = []

    # Repeat the test cycle 10 times
    for i in range(10):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

        # Train the Naive Bayes model
        nb_model = CategoricalNaiveBayes(laplace_smoothing=1)
        nb_model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = nb_model.predict(X_test)

        # Calculate the accuracy
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)

    # Calculate mean accuracy and standard deviation of accuracies
    mean_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)

    # Output the results
    print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")
    print(f"Standard Deviation of Accuracies: {std_deviation * 100:.2f}%")

if __name__ == "__main__":
    main()
