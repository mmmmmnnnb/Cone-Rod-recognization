from gc import callbacks
from os import walk
from os.path import exists
import sys
import numpy
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import random


model = tf.keras.models.load_model('D:/OSU/CSE5194/Cone Rod Manual Process/data/Checkpoint/model')

inputTestFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_TestMatrix/'
labelFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_TestResult/'

#the list of image names
f = []

#get all the images to the list
for (dirpath, dirnames, filenames) in walk(inputTestFolder):
    f.extend(filenames)
    break

for filename in f:
    with open(inputTestFolder + filename, 'rb') as file:
        matrix = numpy.load(file)
        labels = numpy.empty((1, 3))
        for x in range(matrix.shape[0] - 64):
            for y in range(matrix.shape[1] - 64):

                patch = matrix[x:x+65, y:y+65]
                patch = patch/numpy.amax(patch)

                label = model.predict(patch)

                numpy.append(labels, label, axis = 0)

        numpy.save(labelFolder + filename[:len(filename)-8] + '_Labels', labels)
        