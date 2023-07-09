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
