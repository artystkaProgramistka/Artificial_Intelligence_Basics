import csv
import pandas as pd
import pandas as pd
import sqlite3
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


def create_and_show_gui():
    # Create the Tkinter application window
    window = tk.Tk()
    window.title("Wine Classifier")

    train_df, test_df = read_data() # added to load headers for prediction form

    show_train_button = tk.Button(window, text="Show train data", command=lambda: display_data(window, 'train'))
    show_train_button.pack(pady=10)
    show_test_button = tk.Button(window, text="Show test data", command=lambda: display_data(window, 'test'))
    show_test_button.pack(pady=10)

    # # Create a submit button
    train_button = tk.Button(window, text="Train Model", command=train_model)
    train_button.pack(pady=10)

    test_button = tk.Button(window, text="Test Model", command=save_results_and_plots)
    test_button.pack(pady=10)

    add_data_button = tk.Button(window, text="Add Data", command=lambda: add_observation(window)) # lambda added to pass arguments
    add_data_button.pack(pady=10)

    predict_button = tk.Button(window, text="Predict Wine Type", command=lambda: predict_wine(window)) # lambda added to pass arguments
    predict_button.pack(pady=10)

    # reset database button
    train_button = tk.Button(window, text="Reset database", command=reset_database)
    train_button.pack(pady=10)

    # Run the Tkinter event loop
    window.mainloop()


def submit_predict_form(best_model, entries, predict_window):
    # Extract the values from the entry fields
    values = [float(entry.get()) for entry in entries]

    # Use the model to predict the wine type
    predicted_type = best_model.predict([values])

    # Show a message box with the predicted wine type
    messagebox.showinfo("Predicted Wine Type", "The predicted wine type is {}.".format(predicted_type[0]))

    # Destroy the add window
    predict_window.destroy()


def predict_wine(window):
    # Connect to the SQLite database
    conn = sqlite3.connect('wine_data.db')
    cursor = conn.cursor()

    # Get the column names from the database
    cursor.execute("PRAGMA table_info(wine_data_train)")  # Use the correct table name
    headers = [column[1] for column in cursor.fetchall()[1:]]  # Exclude the first column (Type of Wine)

    # Create a new window for the predict form
    predict_window = tk.Toplevel(window)
    predict_window.title("Predict Wine Type")

    # Load the saved model from disk
    best_model = joblib.load('best_model.pkl')

    # Create labels and entry fields for each column
    entries = []
    for col in headers:
        label = tk.Label(predict_window, text=col)
        label.pack()
        entry = tk.Entry(predict_window)
        entry.pack()
        entries.append(entry)

    # The command for the button now has to be a lambda function to pass the arguments
    submit_button = tk.Button(predict_window, text="Submit", command=lambda: submit_predict_form(best_model, entries, predict_window))
    submit_button.pack()

    # Close the database connection
    conn.close()


def read_data():
    headers = ["Type of Wine", "Alcohol", "Malic acid", "Ash", "Alcanity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
    df = pd.read_csv("wine_data.csv", names=headers, dtype={"Type of Wine": int})  # Specify dtype for the label column as int
    train_df, test_df = train_test_split(df, test_size=0.2)
    return train_df, test_df

def save_df_to_database(df, split):
    conn = sqlite3.connect('wine_data.db')
    cursor = conn.cursor()

    # or create one if it does not already exist
    # Create a table to store the data
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS wine_data_{split} (
        "Type of Wine" INTEGER,
        "Alcohol" REAL,
        "Malic acid" REAL,
        "Ash" REAL,
        "Alcanity of ash" REAL,
        "Magnesium" REAL,
        "Total phenols" REAL,
        "Flavanoids" REAL,
        "Nonflavanoid phenols" REAL,
        "Proanthocyanins" REAL,
        "Color intensity" REAL,
        "Hue" REAL,
        "OD280/OD315 of diluted wines" REAL,
        "Proline" REAL
    )
    """
    cursor.execute(create_table_query)

    # Insert the data into the table
    insert_query = f"INSERT INTO wine_data_{split} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    cursor.executemany(insert_query, df.values.tolist())
    conn.commit()

    # Close the connection
    conn.close()


def reset_database():
    # Connect to the SQLite database
    conn = sqlite3.connect('wine_data.db')
    cursor = conn.cursor()

    # Drop the existing table
    cursor.execute("DROP TABLE IF EXISTS wine_data_train")
    cursor.execute("DROP TABLE IF EXISTS wine_data_test")

    # Commit the changes
    conn.commit()

    # Load the data from the CSV file
    train_df, test_df = read_data()

    save_df_to_database(train_df, 'train')
    save_df_to_database(test_df, 'test')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    # Show a message box indicating successful reset
    messagebox.showinfo("Reset the database", "Reset operation was performed successfully!")


def train_model():
    X_train, y_train = read_split('train')

    # Perform a grid search to find the best parameters
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=KFold(n_splits=5, random_state=42, shuffle=True))
    grid_search.fit(X_train, y_train)

    # Save the best model to disk
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'best_model.pkl')

    # Test model after training
    accuracy_test, accuracy_train, confusion_matrix_test, confusion_matrix_train, classification_report_test, classification_report_train = test_model()

    results_df = pd.DataFrame(grid_search.cv_results_)

    # Save results and plot
    plot_cv_results(results_df)

    messagebox.showinfo("Training the model", "Model trained successfully!\nTrain Accuracy: {}\nTest Accuracy: {}".format(
                            accuracy_train, accuracy_test))


def read_split(split):
    # Connect to the SQLite database
    conn = sqlite3.connect('wine_data.db')
    # Query the data from the database
    query = f"SELECT * FROM wine_data_{split}"
    df = pd.read_sql_query(query, conn)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    conn.close()
    return X, y


def display_data(window, split):
    # Connect to the SQLite database
    conn = sqlite3.connect('wine_data.db')

    # Query the data from the database
    query = f"SELECT * FROM wine_data_{split}"
    df = pd.read_sql_query(query, conn)

    # Create a new window for the table display
    table_window = tk.Toplevel(window)
    table_window.title("Data Display")

    # Create a Treeview widget to display the data as a table
    tree = ttk.Treeview(table_window)

    # Add columns to the Treeview
    tree["columns"] = list(df.columns)

    # Format the columns
    tree.column("#0", width=0, stretch=tk.NO)  # Hide the first column (index)
    for col in df.columns:
        tree.column(col, anchor=tk.CENTER, width=100, stretch=tk.NO)
        tree.heading(col, text=col)

    # Insert the data into the Treeview
    for i, row in df.iterrows():
        tree.insert("", tk.END, text=i, values=list(row))

    # Add the Treeview to the window
    tree.pack()

    # Close the database connection
    conn.close()


def submit_form(add_window, entries):
    # Connect to the SQLite database
    conn = sqlite3.connect('wine_data.db')
    cursor = conn.cursor()

    # Extract the values from the entry fields
    values = [entry.get() for entry in entries]

    # Insert the new observation into the csv file
    with open('wine_data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(values)

    train_df, test_df = read_data()
    save_df_to_database(train_df, 'train')
    save_df_to_database(test_df, 'test')

    # Close the database connection
    conn.close()

    # Show a message box indicating successful addition
    messagebox.showinfo("Add Observation", "New observation added successfully!")

    # Destroy the add window
    add_window.destroy()


def add_observation(window):
    # Connect to the SQLite database
    conn = sqlite3.connect('wine_data.db')
    cursor = conn.cursor()

    # Get the column names from the database
    cursor.execute("PRAGMA table_info(wine_data_train)")  # Use the correct table name
    headers = [column[1] for column in cursor.fetchall()]  # Include the first column (Type of Wine)

    # Create a new window for the add observation form
    add_window = tk.Toplevel(window)
    add_window.title("Add Observation")

    # Create labels and entry fields for each column
    entries = []
    for col in headers:
        label = tk.Label(add_window, text=col)
        label.pack()
        entry = tk.Entry(add_window)
        entry.pack()
        entries.append(entry)

    # The command for the button now has to be a lambda function to pass the arguments
    submit_button = tk.Button(add_window, text="Submit", command=lambda: submit_form(add_window, entries))
    submit_button.pack()

    # Close the database connection
    conn.close()


def plot_cv_results(results):
    results.to_csv("results.csv")

    # Visualization and saving plots showing grid search results and fit time
    plt.figure(figsize=(16, 6))

    # Plotting mean test scores
    plt.subplot(1, 2, 1) # plt.subplot(num_rows, num_cols, plot_index)
    sns.lineplot(x='param_n_neighbors', y='mean_test_score', hue='param_metric', data=results)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean Test Score')
    plt.title('Grid Search: Mean Test Score')

    # Plotting fit times
    plt.subplot(1, 2, 2)
    sns.lineplot(x='param_n_neighbors', y='mean_fit_time', hue='param_metric', data=results)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean Fit Time (seconds)')
    plt.title('Grid Search: Mean Fit Time')

    # Save the plot as PNG
    plt.savefig('mean_score_&_fit_time_plot.png')
    plt.show()


def save_results_and_plots():
    accuracy_test, accuracy_train, confusion_matrix_test, confusion_matrix_train, classification_report_test, classification_report_train = test_model()

    # Visualization and saving the confusion matrix as a PNG file
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix_test, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix_train, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Train Set)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()

def test_model():
    X_train, y_train = read_split('train')
    X_test, y_test = read_split('test')

    best_model = joblib.load('best_model.pkl')

    predictions_test = best_model.predict(X_test)
    accuracy_test = accuracy_score(predictions_test, y_test)

    predictions_train = best_model.predict(X_train)
    accuracy_train = accuracy_score(predictions_train, y_train)

    confusion_matrix_train = confusion_matrix(y_train, predictions_train)
    classification_report_train = classification_report(y_train, predictions_train)

    confusion_matrix_test = confusion_matrix(y_test, predictions_test)
    classification_report_test = classification_report(y_test, predictions_test)

    return accuracy_test, accuracy_train, confusion_matrix_test, confusion_matrix_train, classification_report_test, classification_report_train


def main():
    train_df, test_df = read_data()

    # Connect to the database and save the data
    save_df_to_database(train_df, 'train')
    save_df_to_database(test_df, 'test')

    # Create the GUI
    create_and_show_gui()


if __name__ == "__main__":
    main()