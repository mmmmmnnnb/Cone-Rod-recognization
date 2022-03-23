from os import walk
from PIL import Image
import cv2
import numpy
import matplotlib.pyplot as plt


#the folder where store the input images
folder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/Cropped_records_Mengxi/Cropped_records_Mengxi/cropped_imgs/'
#the folder where store the transfered data
saveFolder = 'D:/OSU/CSE5194/Cone Rod Manual Process/data/MatrixLikeImage/'

#the list of image names
f = []

#get all the images to the list
for (dirpath, dirnames, filenames) in walk(folder):
    f.extend(filenames)
    break

#store all the images as matrices
for filename in f:

    	
    im = cv2.imread(folder +  filename, cv2.IMREAD_GRAYSCALE)
    
    #transfer the image to matrix
    imArray = numpy.array(im)
    maxPixel = numpy.max(imArray)
    scale = 255/maxPixel
    imArray = imArray*scale
    #create a matrix which is 100 higher and wider, and all set to 4999 (if a 10*10 block with all its value under 5000, then don't pick it as an input)
    saveArray = numpy.full((imArray.shape[0] + 65, imArray.shape[1] + 65), numpy.mean(imArray)-2*numpy.std(imArray))
    #place the image to the center of the matrix
    saveArray[32:(saveArray.shape[0] - 33), 32:(saveArray.shape[1] - 33)] = imArray
    #store the matrix
    numpy.save(saveFolder + filename, saveArray)




   