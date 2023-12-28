import cv2, numpy, os
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dir, files) in os.walk(datasets) :
    
