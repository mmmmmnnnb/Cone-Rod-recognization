from gc import callbacks
from os import walk
from os.path import exists
import sys
import numpy
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import random


def create_model():
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (5, 5), strides=1, input_shape=(65, 65, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(layers.ReLU())
    model.add(layers.Conv2D(32, (5, 5), strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.AveragePooling2D(pool_size = (3,3), strides = 2))
    model.add(layers.Conv2D(64, (5, 5), strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.AveragePooling2D(pool_size = (3,3), strides = 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(3))
    model.add(layers.Softmax())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



callBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=5, verbose=0, mode='min', baseline=None, restore_best_weights=False)
    
inputTrainFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_inputTrain/'
labelTrainFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_LabelTrain/'
inputTestFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_inputTest/'
labelTestFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_LabelTest/'

inputTrainFiles = []
labelTrainFiles = []
inputTestFiles = []
labelTestFiles = []

inputArrTrain = numpy.empty((1, 65, 65))
labelArrTrain = numpy.empty((1, 3))
inputArrTest = numpy.empty((1, 65, 65))
labelArrTest = numpy.empty((1, 3))

for (dirpath, dirnames, filenames) in walk(inputTrainFolder):
    inputTrainFiles.extend(filenames)
    break

for (dirpath, dirnames, filenames) in walk(labelTrainFolder):
    labelTrainFiles.extend(filenames)
    break

for (dirpath, dirnames, filenames) in walk(inputTestFolder):
    inputTestFiles.extend(filenames)
    break

for (dirpath, dirnames, filenames) in walk(labelTestFolder):
    labelTestFiles.extend(filenames)
    break



for filename in inputTrainFiles:
    with open(inputTrainFolder + filename, 'rb') as file:
        patches = numpy.load(file)
        inputArrTrain = numpy.append(inputArrTrain, patches, axis = 0)

for filename in labelTrainFiles:
    with open(labelTrainFolder + filename, 'rb') as file:
        labels = numpy.load(file)
        labelArrTrain = numpy.append(labelArrTrain, labels, axis = 0)

for filename in inputTestFiles:
    with open(inputTestFolder + filename, 'rb') as file:
        patches = numpy.load(file)
        inputArrTest = numpy.append(inputArrTest, patches, axis = 0)

for filename in labelTestFiles:
    with open(labelTestFolder + filename, 'rb') as file:
        labels = numpy.load(file)
        labelArrTest = numpy.append(labelArrTest, labels, axis = 0)

model = create_model()

model.summary()


randomPoint= random.randint(0,len(inputArrTrain)-1)

#matrix = inputArrTrain[randomPoint]
#label = labelArrTrain[randomPoint]
#plt.imshow(matrix)
#plt.show()
#input()




model.fit(inputArrTrain, labelArrTrain,  epochs=50, validation_split = 0.2, callbacks = callBack)

inputArrTrain = []

#model.save('D:/OSU/CSE5194/Cone Rod Manual Process/data/Checkpoint/model')

inputTestFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_TestMatrix/'
labelFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_TestResult/'

#the list of image names
f = []

#get all the images to the list
for (dirpath, dirnames, filenames) in walk(inputTestFolder):
    f.extend(filenames)
    break

for filename in f:
    if not exists(labelFolder + filename[:len(filename)-8] + '_Labels.npy'):
        with open(inputTestFolder + filename, 'rb') as file:
            matrix = numpy.load(file)
            labels = []
            for x in range(matrix.shape[0] - 64):
                shape0 = []
                for y in range(matrix.shape[1] - 64):

                    patch = numpy.array([matrix[x:x+65, y:y+65],])
                    patch = patch/numpy.amax(patch)

                    label = model.predict(patch)
                    a = [0] * 3
                    a[numpy.argmax(label)] = 1

                    shape0.append(a)
                labels.append(shape0)

            numpy.save(labelFolder + filename[:len(filename)-8] + '_Labels', numpy.array(labels))