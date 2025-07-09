import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = [
    "sepal_length(cm)",
    "sepal_width(cm)",
    "petal_length(cm)",
    "petal_width(cm)",
    "Iris-species",
]
df = pd.read_csv(url, names=cols)

print(df)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

class_mv = {}

for cls in np.unique(y_train):
    class_data = X_train[y_train == cls]
    class_mv[cls] = {
        "mean": np.mean(class_data, axis=0),
        "var": np.var(class_data, axis=0),
        "prior": len(class_data) / len(X_train),
    }
    print(f"\nClass: {cls}\n", class_data)


def Prob_density(x, mean, var):
    eps = 1e-6
    coeff = 1 / np.sqrt(2 * np.pi * (var + eps))
    exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
    return coeff * exponent


def predict(sample):
    posteriors = {}
    for cls, items in class_mv.items():
        likelihoods = Prob_density(sample, items["mean"], items["var"])
        total_likelihood = np.prod(likelihoods)
        posteriors[cls] = total_likelihood * items["prior"]
    return max(posteriors, key=posteriors.get)


sample = X_test[0]
predicted_class = predict(sample)
print("\nPredicted(testing):", predicted_class)
print("Actual(testing)   :", y_test[0])


random_index = random.randint(0, len(X_test) - 1)
sample = X_test[random_index]
true_label = y_test[random_index]

predicted = predict(sample)
print("\nSample values  :", sample)
print("Predicted class(Testing):", predicted)
print("Actual class(given Testing)   :", true_label)


custom_sample = np.array([6.9, 3.1, 5.4, 2.1])
predicted_class = predict(custom_sample)
print("\nCustom Sample     :", custom_sample)
print("Predicted Class   :", predicted_class)


y_pred = [predict(x) for x in X_test]
correct = sum(p == t for p, t in zip(y_pred, y_test))
accuracy = correct / len(y_test)

print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


colors = {"Iris-setosa": "red", "Iris-versicolor": "green", "Iris-virginica": "blue"}
