from os import walk
from os.path import exists
import numpy
import csv
import matplotlib.pyplot as plt
import math

labelFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_Label/'

f = []

#get all the images to the list
for (dirpath, dirnames, filenames) in walk(labelFolder):
    f.extend(filenames)
    break

ConeSample = 0
RodSample = 0
BackGroundSample = 0

for filename in f:
    with open(labelFolder + filename, 'rb') as file:
        matrix = numpy.load(file)
    for label in matrix:
        if label[0] == 1:
            ConeSample += 1
        elif label[1] == 1:
            RodSample += 1
        elif label[2] == 1:
            BackGroundSample += 1

print('ConeSample= ' + str(ConeSample))
print('RodSample= ' + str(RodSample))
print('BackGroundSample= ' + str(BackGroundSample))