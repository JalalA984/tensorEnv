import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

df = pd.read_csv("LinRegression/student-mat.csv", sep=";")

# These are attributes (unique to each "instance"/student) except for G3 which will be removed when test/train
data = df[["G1","G2", "G3", "studytime", "failures","health", "absences", "age"]]

# Our "label" is our output, in other words, based on the above attributes we want to predict the label "G3"
predict = "G3"

# Array of attributes removing G3
x = np.array(data.drop([predict], axis=1))
# Array of our label
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

'''
best = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    accu = linear.score(x_test,y_test)

    if accu > best:
        best = accu
        print(best)
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''

pickle_in = open("LinRegression/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: ", linear.coef_)
print("intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()