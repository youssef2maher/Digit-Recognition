from keras.datasets import mnist
import matplotlib.pyplot as plot
import numpy as npy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import math

# storing the mnist in two train arrays and two test arrays
# the x arrays carry the images and the y arrays carry the values of the images
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
- This function splits the image into grids.
- It takes the image and the grids number as parameters
  and check if the grids number could cause a problem.
  if it causes a problem of splitting uneven grids, it handles it.
- Then, it stores the splitted grids in grids list and return that list.
"""


def split(image, grids_number):
    # Calculating the dimension of the grid
    pixels_number = math.ceil(28 / grids_number)
    image_cpy = image

    # To handle the uneven grids problem
    if (28 % grids_number != 0):

        new_size = pixels_number * grids_number
        image_cpy = npy.zeros([new_size, new_size])

        for k in range(0, len(image)):
            for o in range(0, len(image[0])):
                image_cpy[k][o] = image[k][o]

    # The grids list and its counter to store the grids in it
    grids_index = 0
    grids = [0 for i in range(0, grids_number * grids_number)]

    # Splitting and storing the grids into the grids list
    for i in range(0, grids_number):
        for j in range(0, grids_number):
            grids[grids_index] = image_cpy[i * pixels_number:(i + 1) * pixels_number,
                                 j * pixels_number:(j + 1) * pixels_number]
            grids_index += 1

    return grids


"""
- This function calculates the centroid of an image grids.
- It takes an image grids list and the grids_number as a parameter.
- Then, it calculates the centroid of each grid and put the values (Xc and Yc)
  of each grid in the feature vector list and return that vector.
"""


def centroid(grids, grids_number):
    # Calculating the dimension of the grid and creating the feature vector
    pixels_number = math.ceil(28 / grids_number)
    feature_vector = [0 for i in range(0, grids_number * grids_number * 2)]
    Xc = 0
    Yc = 0

    # Calculating the centroid of each grid
    for i in range(0, grids_number * grids_number):

        pixels_sum = 0
        for j in range(0, pixels_number):
            for k in range(0, pixels_number):
                Xc += grids[i][j][k] * (j + 1)
                Yc += grids[i][j][k] * (k + 1)
                pixels_sum = pixels_sum + grids[i][j][k]

        if (pixels_sum != 0):
            feature_vector[2 * i] = Xc / pixels_sum
            feature_vector[2 * i + 1] = Yc / pixels_sum

        else:
            feature_vector[2 * i] = 0
            feature_vector[2 * i + 1] = 0

    return feature_vector


"""
- This function predict the digit value by using the nearst neighbour algorithm
- It takes the grids number as a paramter.
- It splits the images existed in the train and the test datasets into grids 
  and gets the centroid value of each grid and stores the values in two lists
  a list for the train dataset images and another for the test ones.
- Then, it uses the built-in classifier to fit and predict the two sets.
- Finally, it calculates the accuracy of the predection and return its value.
"""


def nearest_neighbour(grids_number):
    # Splitting the image into grids and calculating and storing the centroid
    # for the train images
    fv_train = npy.zeros([len(x_train), grids_number * grids_number * 2])
    for i in range(len(x_train)):
        grids = split(x_train[i], grids_number)
        fv_train[i] = centroid(grids, grids_number)

    # Splitting the image into grids and calculating and storing the centroid
    # for the test images
    fv_test = npy.zeros([len(x_test), grids_number * grids_number * 2])
    for i in range(len(x_test)):
        grids = split(x_test[i], grids_number)
        fv_test[i] = centroid(grids, grids_number)

    # Using the calssifier to fit the train dataset images
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(fv_train, y_train)

    # Predicting the value of the test dataset images
    prediction = model.predict(fv_test)

    # Calculating the accuracy of the prediction operation
    accuracy = accuracy_score(y_test, prediction)

    return accuracy


accuracy = nearest_neighbour(4)
print("Accuracy: ", accuracy)
