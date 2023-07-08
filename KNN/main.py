import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

data = pd.read_csv("car.data", sep=",")

# Preprocessing from sklearn can help turn features to numeric ints
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''
best = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # KNN Algo model
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)

    if acc > best:
        best = acc
        print(best)
        with open("carmodel.pickle", "wb") as f:
            pickle.dump(model, f)
'''

pickle_in = open("carmodel.pickle", "rb")
model = pickle.load(pickle_in)

print("accuracy based on current test datasets: ", model.score(x_test,y_test))

predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    #print(model.kneighbors([x_test[x]]), 7)
