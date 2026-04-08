import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print("Machine Learning - Example 1")

iris = load_iris()

print("Features: columns are sepal length, sepal width, petal length, petal width\n",iris.data)
print("labels: 0, 1, 2 correspond to the three iris species\n",iris.target)

print("Splitting into training and testing sets")
print("Usually: 80% train, 20% test")
X = iris.data
y = iris.target

print("Building a k-NN Classifier")
test_size=0.2
random_state=42
print("Controls the shuffling applied to the data2 before applying the split. Pass an int for reproducible output across multiple function calls.")

n_neighbors=3

print(f"Selected: {(1 - test_size) * 100}% train, {test_size * 100}% test")
print(f"Selected random state: {random_state}")
print(f"Selected n neighbors: {n_neighbors}")
print("Initialize a k-NN classifier and train it")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)  # learn from the training data2
print(model)

print("Make predictions on the test set")
y_pred = model.predict(X_test)
print(y_pred)

# 5. Evaluate the accuracy of the model on the test set
accuracy = (y_pred == y_test).mean()
print(f"Test accuracy: {accuracy}")
