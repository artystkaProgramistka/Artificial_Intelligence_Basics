# Multinomial Naive Bayess
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class GaussianNaiveBayes:
    def __init__(self, laplace_smoothing=1):
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.priors = class_counts / len(y)

        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = np.mean(X_c, axis=0)
            self.var[i, :] = np.var(X_c, axis=0) + self.laplace_smoothing

    def predict(self, X):
        likelihoods = np.zeros((X.shape[0], len(self.classes)))

        for i, c in enumerate(self.classes):
            likelihoods[:, i] = multivariate_normal.pdf(X, mean=self.mean[i], cov=np.diag(self.var[i]))

        posteriors = likelihoods * self.priors
        predictions = np.argmax(posteriors, axis=1)

        return self.classes[predictions]

class MultinomialNaiveBayes:
    def __init__(self, laplace_smoothing=1):
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.likelihoods = {}
        self.priors = {}
        self.num_rows_given_label = {}  # Store the number of rows for each class

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.num_rows_given_label[c] = X_c.shape[0]  # Store the number of rows for this class
            self.likelihoods[c] = {}
            for feature_index in range(X_c.shape[1]):
                self.likelihoods[c][feature_index] = {}
                unique_feature_values, feature_counts = np.unique(X_c[:, feature_index], return_counts=True)
                for value, count in zip(unique_feature_values, feature_counts):
                    self.likelihoods[c][feature_index][value] = (count) / (
                                np.sum(feature_counts) + len(unique_feature_values))

    def predict(self, X):
        posteriors = {}
        for c in self.classes:
            likelihood_values = np.zeros_like(X).astype(float)
            for feature_index in range(X.shape[1]):
                val_likelihoods = self.likelihoods[c][feature_index]

                # Check which feature values are available in the likelihoods
                likelihood_available = np.isin(X[:, feature_index], list(val_likelihoods.keys()))

                # Use np.vectorize to calculate the likelihoods for available feature values
                get_val_likelihood = np.vectorize(val_likelihoods.get)
                likelihood_values[likelihood_available, feature_index] = get_val_likelihood(
                    X[likelihood_available, feature_index])

                likelihood_values[~likelihood_available, feature_index] = self.laplace_smoothing / (
                        len(val_likelihoods) * self.laplace_smoothing + self.num_rows_given_label[c])

            posteriors[c] = self.priors[c] + np.sum(np.log(likelihood_values), axis=1)

        predictions = []
        for i in range(len(X)):
            sample_posteriors = [posteriors[c][i] for c in self.classes]
            predictions.append(self.classes[np.argmax(sample_posteriors)])
        return predictions


def main():
    #########################
    # Gaussian Naive Bayess #
    #########################
    # Load the data
    train_data = np.loadtxt("data_iris/train.txt", delimiter=",", dtype=object)
    test_data = np.loadtxt("data_iris/test.txt", delimiter=",", dtype=object)

    # Split the data into input features (X) and labels (y)
    X_train, y_train = train_data[:, :-1].astype(float), train_data[:, -1]
    X_test, y_test = test_data[:, :-1].astype(float), test_data[:, -1]

    # Train the Naive Bayes model
    nb_model = GaussianNaiveBayes()
    nb_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = nb_model.predict(X_test)

    # Calculate the accuracy
    accuracy = np.mean(y_pred == y_test)
    print('Gaussian on Iris_data')
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    unique_labels = np.unique(np.concatenate((y_train, y_test)))
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for true_label, pred_label in zip(y_test, y_pred):
        i = np.where(unique_labels == true_label)[0][0]
        j = np.where(unique_labels == pred_label)[0][0]
        confusion_matrix[i, j] += 1

    print("\nConfusion matrix:")
    print(confusion_matrix)

    ############################
    # Multinomial Naive Bayess #
    ############################
    # Load the data
    train_data = np.loadtxt("data_car/train.txt", delimiter=",", dtype=object)
    test_data = np.loadtxt("data_car/test.txt", delimiter=",", dtype=object)

    # Convert numpy array to DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Initialize the encoders
    feature_encoders = [LabelEncoder() for _ in range(train_df.shape[1] - 1)]
    label_encoder = LabelEncoder()

    # Apply label encoding to each feature column
    for column in range(train_df.shape[1] - 1):
        train_df[column] = feature_encoders[column].fit_transform(train_df[column])
        test_df[column] = feature_encoders[column].transform(test_df[column])

    # Apply label encoding to the label column
    train_df[train_df.shape[1] - 1] = label_encoder.fit_transform(train_df[train_df.shape[1] - 1])
    test_df[test_df.shape[1] - 1] = label_encoder.transform(test_df[test_df.shape[1] - 1])

    # Convert DataFrame back to numpy array
    train_data = train_df.to_numpy()
    test_data = test_df.to_numpy()

    # Split the data into input features (X) and labels (y)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # Train the Naive Bayes model
    nb_model = MultinomialNaiveBayes()
    nb_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = nb_model.predict(X_test)

    # Calculate the accuracy
    accuracy = np.mean(y_pred == y_test)
    print('Multinomial on Car_data')
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    unique_labels = np.unique(np.concatenate((y_train, y_test)))
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for true_label, pred_label in zip(y_test, y_pred):
        i = np.where(unique_labels == true_label)[0][0]
        j = np.where(unique_labels == pred_label)[0][0]
        confusion_matrix[i, j] += 1

    print("\nConfusion matrix:")
    print(confusion_matrix)

if __name__ == "__main__":
    main()

