# we use two alogs here, which are Fisher Face algo and LBPH(local border pattern) algo

# # workflow of face recognition
# 1.  Loading face detection algo --> 2. Loading classifier for face recognition 
# 3. Training classifier for our dataset --> 4. Reading farme from camera & pre-processing
# 5. face detection by its algo ---> 6. predicting face by loading frame into model --> 7. displays recongnized class with its accuracy

import cv2, os
haar_file = 'haarcascade_frontalface_default.xml'

datasets = 'datasets' # create folder name 'datasets' in current directory !
# give diff names for sub_data and put their face image infront of camera till it captures 30 images and keep on doing it till 30 people !! 
sub_data = 'steve' # file name and the data of which person we are collecting

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path) :
    os.mkdir(path)
(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

count = 1
while count < 31 :
    print(count)
    (_, img) = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces : 
        cv2.rectangle(img , (x,y), (x + w, y + h), (0, 0, 255), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count += 1
    
    cv2.imshow('OpenCv', img)
    key = cv2.waitKey(10)
    if key == 27 : # esc key
        break

webcam.release()
cv2.destroyAllWindows()