import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data from the CSV file
file_path = 'car_evaluation.data'
cars_data = pd.read_csv(file_path, header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptance'])

# Convert categorical data to numerical data using LabelEncoder
le = LabelEncoder()
for column in cars_data.columns:
    cars_data[column] = le.fit_transform(cars_data[column])

# Define features (X) and target (y)
X = cars_data.drop('acceptance', axis=1)
y = cars_data['acceptance']

# Placeholder for accuracies
accuracies = []

# Perform the split and testing 10 times
for i in range(10):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Initialize and train the Naive Bayes classifier
    nb_classifier = CategoricalNB()
    nb_classifier.fit(X_train, y_train)

    # Predict the test set results
    y_pred = nb_classifier.predict(X_test)

    # Calculate and store the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Calculate mean accuracy and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_deviation = np.std(accuracies)

# Output the results
print(mean_accuracy, std_deviation)
