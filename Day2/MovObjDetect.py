import cv2
import time
import imutils

cam = cv2.VideoCapture(0) # video capture object
time.sleep(1)

# better to make the firstFrame a full static image to observe the results better !! 
firstFrame = None # initializing there is no obj/specifically no img frame is there till the loop is run once  
area = 500 # threshold for the contour area, if the area is less than 500, the contour obj is ignored 

while True: 
    _, img = cam.read() # place the frame in img as the first arg is just not needed(which is whether the frame is successfully read or not - 1 or 0 is returned)
    text = "Normal" # if the image is static -- no moving objects, then normal state
    
    # we resize, convert to grayscale inorder to convert to gaussian image, gaussian is needed as a smooth frame is needed for optimization and better performance and better segment identification 
    img = imutils.resize(img, width=500) 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
    
    # The concept is to subtract the current image from the fundamental frame captured in the first instance.. here it is firstFrame ..
    # The if condition is only run once to take the first frame and never again and continue's the loop
    if firstFrame is None: 
        firstFrame = gaussianImg
        continue
    
    # The absolute diff is obtained by subtracting the current image from the firstFrame .. that is called 'movement observed' ! 
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] # done to obtain a pure black & white img - to observe and process better.. 

    # after thresholding and difference finding... there will be some gaps in the img -> they are refilled by dilating with the border pixel values and iterations would be doing the process twice !
    threshImg = cv2.dilate(threshImg, None, iterations=2) 

    #  -- IMP PART --
    #  contours are basically the moving obj recognized .. we need to take them all from one frame and an array of contours in made.. then make a green rectangle around them for us to see !!  
    #

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,  # here img's copy is made, then only 'external' contours are 'retrieved'(means a contour inside a contour is ignored)
                            cv2.CHAIN_APPROX_SIMPLE) # this means to connect the chain of contours that share the same boundary and show it in as one big rectangle 
    cnts = imutils.grab_contours(cnts) # making it as array and to ensure compatibilty with other openCV versions
    
    for c in cnts :  # removing the contours and thresholding them by putting a limit
        if cv2.contourArea(c) < area : 
            continue
        (x, y, h, w) = cv2.boundingRect(c) # boundary of that single contour is taken (origin x, origin y, height, width)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # rectangles is made with img as input/src, orgin, extreme point, color of border and thickness, these are passed as params
        text = "Moving object detected" # display this 
    print(text)
    
    #
    # Usually an unneccesary step to show the text on the video capture part if it is normal or moving !! 
    cv2.putText(img, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # this are the properties of text like (height and width of text, fond style, font scale,  color and thickness)
    cv2.imshow("Camera Feed", img) # show/display the feed to the user 
    
    key = cv2.waitKey(1) & 0xFF # quit if 'q' is pressed !! only to take 8 bits and process it perfecly, it is 'and' with FF(8 highs)
    if(key == ord('q')) : 
        break

cam.release() # release the start, if not the cam keeps on staying up !! 
cv2.destroyAllWindows()

