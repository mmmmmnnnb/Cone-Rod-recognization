from os import walk
from os.path import exists
from turtle import position
import matplotlib.pyplot as plt
import numpy
import csv
import math
import random


def determinable(centerMatrix, minBright):
    return numpy.any(centerMatrix > minBright)

def getBGRegion(record, matrix):
    
    acceptablepoints = []
    for x in range(matrix.shape[0] - 64):
        for y in range(matrix.shape[1] - 64):
            accept = True
            for row in record:
                distance = math.sqrt( (row[1] - x)**2 + (row[0] - y)**2 )
                if row[2] == 1:
                    if distance < 10:
                        accept = False
                elif row[2] == 2:
                    if distance < 5:
                        accept = False
            if accept:
                acceptablepoints.append([x,y])
    return acceptablepoints


#def generateConeRod(record, matrix, smallPatches, labels, maxSize):

def generateConeRod(record, matrix, smallPatches, labels):
    i = 0
    for row in record:

        x = row[1]
        y = row[0]

        if matrix[x:x+65, y:y+65].shape != (65, 65):
            continue
        if row[2] == 1:
            labels.append(numpy.array([1, 0, 0]))
            smallPatches.append(matrix[x:x+65, y:y+65]/numpy.amax(matrix[x:x+65, y:y+65]))

           
            

        elif row[2] == 2:
            labels.append(numpy.array([0, 1, 0]))
            smallPatches.append(matrix[x:x+65, y:y+65]/numpy.amax(matrix[x:x+65, y:y+65]))

        i = i+1


            



#the folder where store the transfered data
saveFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/MatrixLikeImage/'
recordFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/Cropped_records_Mengxi/Cropped_records_Mengxi/manual_coords/'

#the folder where store the input and labels
inputFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_inputTest/'
labelFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_LabelTrain/'

#the list of image names
f = []

#get all the images to the list
for (dirpath, dirnames, filenames) in walk(saveFolder):
    f.extend(filenames)
    break


for filename in f:

    #if not exists(labelFolder + filename[:len(filename)-8] + '_Labels.npy'):
        record = numpy.loadtxt(recordFolder + filename[:len(filename)-8] + '.txt', dtype='int')
        with open(saveFolder + filename, 'rb') as file:
            matrix = numpy.load(file)
        
            smallPatches = []
            labels = []
                #run the small matrix on the big matrix, find the center target matrix and check it whether it is determinable.

            generateConeRod(record, matrix, smallPatches, labels)

            #patches = numpy.array(smallPatches)

            cellNum = len(labels)

            BGPoints = getBGRegion(record, matrix)

            #for point in BGPoints:

            #    matrix[point[0]+32, point[1]+32] = 0
            #plt.imshow(matrix)
            #plt.show()
            #input()

            index = 0
            while index < cellNum:
            
                randomPoint= random.randint(0,len(BGPoints)-1)

                selctedPoint = BGPoints[randomPoint]
                centerMatrix = matrix[selctedPoint[0]+27: selctedPoint[0]+38, selctedPoint[1]+27: selctedPoint[1]+38]

                if determinable(centerMatrix, matrix[0,0]):
                    patch = matrix[selctedPoint[0]:selctedPoint[0]+65, selctedPoint[1]:selctedPoint[1]+65]
                    smallPatches.append(patch/numpy.amax(patch))
                    labels.append(numpy.array([0, 0, 1]))
                index += 1

        

            numpy.save(inputFolder + filename[:len(filename)-8] + '_Patches', numpy.array(smallPatches))
            numpy.save(labelFolder + filename[:len(filename)-8] + '_Labels', numpy.array(labels))


