# Geovanny Casian
# Spring 2023 CPSC 481-04 1375
# Project 3

import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        evidence = []
        labels = []

        for row in reader:
            evidence.append([
                int(row[0]), float(row[1]), int(row[2]), float(row[3]), int(row[4]),
                float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]),
                int("Jan Feb Mar Apr May June Jul Aug Sep Oct Nov Dec".split().index(row[10])),
                int(row[11]), int(row[12]), int(row[13]), int(row[14]),
                int(row[15] == "Returning_Visitor"), int(row[16] == "TRUE")
            ])
            labels.append(int(row[17] == "TRUE"))

    return evidence, labels


def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    true_positive = sum((labels[i] == predictions[i] == 1) for i in range(len(labels)))
    true_negative = sum((labels[i] == predictions[i] == 0) for i in range(len(labels)))

    sensitivity = true_positive / sum(x == 1 for x in labels)
    specificity = true_negative / sum(x == 0 for x in labels)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
