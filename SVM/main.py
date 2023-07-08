import sklearn
from sklearn import datasets
from sklearn import svm

# Cancer (classification) dataset: features point to tumor that is cancerous or not
cancer = datasets.load_breast_cancer()

# Features and labels
features = cancer.feature_names
labels = cancer.target_names

x = cancer.data
y = cancer.target

# Split data into train and test groups
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["malignant", "benign"]