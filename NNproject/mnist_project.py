import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Upload the MNIST dataset

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the data (scaling to the range of 0-1)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Building the model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Input layer
model.add(tf.keras.layers.Dense(128, activation='relu'))  # First hidden layer
model.add(tf.keras.layers.Dense(128, activation='relu'))  # Second hidden layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Outpu layer

# Compiling the model

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model

model.fit(x_train, y_train, epochs=10)

# Save the model
model.save('handwritten_model.keras')

# Load the model after saving it for later use
model = tf.keras.models.load_model('handwritten_model.keras')


image_number = 0
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except:
        print("error!")
    finally:
        image_number += 1

