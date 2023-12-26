# Object tracking using color using openCV, another way is to use deep learning to do this - which is out of scope for now.
# install pyautogui : open  cmd -> pip install pyautogui

# Object Tracking --------
# Object detection and tracking are the tasks that is imp and challenging such as video survillance and vehicle navigation 
# Image processing is a method of extracting same useful information by converting image into digital inform by performing some operations on it.

# HSV Explanation --------
# HSV Value is designed in the 1970's by computer graphics researchers to more closely align with teh way human vision percieves color-making attributes
# HSV color space (Hue, Saturation, value), its almost like how people select color form color palette and their choice and its with ..
# .. way human choose it and expierince it rather than systematic RGB color space !! 

## Wrok Flow 
# 1. reading frame from camera  ---> 2. pre-processing image
# 3. Finding contours ---> 4. Drawing Minimum enclosing circle
# 5. Finding center of contour area ---> 6. Drawing circle and center ----> 7. Direction based on radius & positon

import imutils
import cv2

GreenLower = (15, 164, 162)
GreenUpper = (45, 232, 255)
camera = cv2.VideoCapture(0)

while True:
    (grabbed, img) = camera.read()
    img = imutils.resize(img, width=600)

    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, GreenLower, GreenUpper)
    mask= cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key = cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        if radius > 10: 
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), -1)
            cv2.circle(img, center, 5, (0, 0, 255), -1)

            if radius > 250:
                print("stop")
            else :
                if center[0] < 150 :
                    print("left")
                elif center[0] > 450 : 
                    print("right") 
                elif radius < 250 :
                    print("front")
                else :
                    print("stop")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

camera.release()
cv2.destroyAllWindows() 

