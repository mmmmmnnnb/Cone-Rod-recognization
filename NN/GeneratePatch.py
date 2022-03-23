from os import walk
from os.path import exists
import numpy
import csv
import matplotlib.pyplot as plt
import math


def determinable(centerMatrix):
    return numpy.any(centerMatrix > 5000 )

def generateLabel(record, centerPoint):
    for row in record:
        distance = math.sqrt( (row[0] - centerPoint[0])**2 + (row[1] - centerPoint[1])**2 )
        if row[2] == 1:
            if distance < 10:
                return [1, 0, 0]
        elif row[2] == 2:
            if distance < 5:
                return [0, 1, 0]
    
    return [0, 0, 1]


#the folder where store the transfered data
saveFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/MatrixLikeImage/'
recordFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/manual_coords/'

#the folder where store the input and labels
inputFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_input/'
labelFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/NN_Label/'

#the list of image names
f = []

#get all the images to the list
for (dirpath, dirnames, filenames) in walk(saveFolder):
    f.extend(filenames)
    break


for filename in f:
    if exists(labelFolder + filename[:len(filename)-8] + '_Labels.npy'):
        record = numpy.loadtxt(recordFolder + filename[:len(filename)-8] + '.txt', dtype='int')
        with open(saveFolder + filename, 'rb') as file:
            matrix = numpy.load(file)
        smallPatches = []
        labels = []
            #run the small matrix on the big matrix, find the center target matrix and check it whether it is determinable.
        for x in range(matrix.shape[0] - 64):
            for y in range(matrix.shape[1] - 64):
                #the section of image we need to view
                viewMatrix = matrix[x:x+65, y:y+65]
                #the center of image we need to figure out whether it is cone, rod, background
                    
                centerMatrix = matrix[x+27: x+38, y+27: y+38]

                #X and Y in the matrix are inverse in manual record.
                #the center point of the original image
                centerPoint = [y, x]
                #if all pixels in the matrix are dimmer than 5000 then it is not determinable.
                if determinable(centerMatrix):
                    label = generateLabel(record , centerPoint)

                    if label == [1,0,0]:
                        matrix[centerPoint[1]+32, centerPoint[0]+32] = 20000
                    elif label == [0,1,0]:
                        matrix[centerPoint[1]+32, centerPoint[0]+32] = 10000
                    else:
                        matrix[centerPoint[1]+32, centerPoint[0]+32] = 5000
                    #smallPatches.append(viewMatrix)
                    #labels.append(generateLabel(record , centerPoint))

                #if x == 86 and y == 20:
        plt.imshow(matrix)
        plt.show()
        input()

        
        numpy.save(inputFolder + filename[:len(filename)-8] + '_Patches', smallPatches)
        numpy.save(labelFolder + filename[:len(filename)-8] + '_Labels', labels)


