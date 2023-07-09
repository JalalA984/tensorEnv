import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Cancer (classification) dataset: features point to tumor that is cancerous or not
cancer = datasets.load_breast_cancer()

# Features and labels
features = cancer.feature_names
labels = cancer.target_names

x = cancer.data
y = cancer.target

for _ in range(10):
    # Split data into train and test groups
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    classes = ["malignant", "benign"]

    # Our classifier
    clsfier = svm.SVC(kernel="linear", C=2)

    # Train data
    clsfier.fit(x_train, y_train)

    y_pred = clsfier.predict(x_test)

    # Test accuracy of our predictions to actual label results
    acc = metrics.accuracy_score(y_test, y_pred) # changes each time program is run
    print(acc)