# The algorithm that we use here is "HAAR CASCADE FRONTAL'FACE' ALGORITHM", which is especially for detecting face only... if for 'car' that is diff algo ! 
# NO RECOGNITION is done !!
# accuracy wise [< Deep Learning ], its easy to do it this way as we are using some others model .. not made by us !! 

# # Algo : 
# - It is based on the haar wavelet technique to analyse pixels in the image into sqauares by function
# - this uses ML techniques to get a high degree of accuracy from what is call 'training data'
# - this uses "integral image" concepts to compute the features detected
# - haar cascade use the adaboost learning algo which selects a small number of imp features from a large set to give an efficient result of classfiers 

# # Workflow : 
# 1. loading "haar cascade frontal face Model file" --> 2. Initializing camera --
# --> 3. reading Frame from camera --> 4. Converting color image into grayscale image --
# --> 5. Obtaining face coordinates by passing algorithm --> 6. Drawing Rectangle on the Face coorinates --
# --> 7. Display the Output Frame

### 1. If you are using a model, we need to do the same pre-processing how the dev guys and what data/image is fed to train the model.. the same way we need to do as well 

import cv2 
alg = "haarcascade_frontalface_default.xml" # path of the haar cascade frontalface xml model file which is a pre-trained model for obj detection

haar_cascade = cv2.CascadeClassifier(alg) # initialises the haar_acscade obj where it takes in the path of the xml file of trained model

cam = cv2.VideoCapture(0) # initialises the video capture  

while True:

    _, img = cam.read() # read the frames from the video ! 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image to grayscale image format since the cascade classifier typically works on gray img, this step is necessary 

    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4) # it returns a list of rectangles each having (x, y, h, w) paramaters to them to display it around the face in furthur process 
    # here 1.3 is a scale factor that tells how much the image size is reduced at each scale.. the smaller value -> detect small faces --> more processing time and ..
    
    # Detailed explanation of why 4 is used is written below ... but if u don't need all that, its just higher number means - that many number of ...
    # .. validations are done in order to recognize is as a face.. so having smaller number might include false positives(face like objs - we don't want them)

    for (x, y, h, w) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cv2.imshow("FaceDetections", img)
    key = cv2.waitKey(10) 
    if key == 27 : # esc key's code is 27  
        break
cam.release()
cv2.destroyAllWindows()

# # Imp notes :: 

# In the context of face detection, a false positive refers to a situation where the algorithm incorrectly identifies an area as a 
# face when there is no actual face present. False positives can occur due to various reasons, such as similar patterns or shapes in 
# the image that resemble faces.

# To reduce false positives, the Haar cascade classifier uses a technique called the Viola-Jones algorithm. This algorithm scans the 
# image at multiple scales and applies a set of predefined features to determine whether each region of the image contains a face or not.

# During this process, the algorithm evaluates each potential detection by considering neighboring rectangles that overlap or are close to 
# each other. The minimum number of neighboring rectangles parameter, often denoted as minNeighbors, specifies the minimum number of nearby
# rectangles required for a detection to be considered valid. If a potential detection does not have enough neighboring rectangles, 
# it is discarded as a false positive.

# By setting a higher value for minNeighbors, the algorithm becomes more conservative in its detection and requires stronger evidence 
# (more neighboring rectangles) to confirm a face. This helps filter out isolated false positive detections that do not have sufficient 
# supporting evidence. However, setting a higher minNeighbors value can also lead to missing some true positive faces, especially in
# complex or crowded scenes.

# The choice of the minNeighbors parameter depends on the specific application and the desired trade-off between reducing false positives 
# and correctly detecting faces. It is typically adjusted through trial and error to achieve a balance that suits the requirements of the 
# task at hand.

# In summary, the minNeighbors parameter in face detection with Haar cascades helps in reducing false positives by considering the 
# presence of neighboring rectangles when determining the validity of a detection.
