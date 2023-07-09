import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load in dataset
data = keras.datasets.fashion_mnist

# Split data into testing and training
(train_images, train_labels), (test_images, test_labels) = data.load_data()

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scaling down data
train_images = train_images/255.0
test_images = test_images/255.0

# Setting up model and neural network

# define layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

# Using our model
prediction = model.predict(test_images)

for x in range(5):
    plt.grid(False)
    plt.imshow(test_images[x], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + labels[test_labels[x]])
    plt.title("Prediction: " + labels[np.argmax(prediction[x])])
    plt.show()
