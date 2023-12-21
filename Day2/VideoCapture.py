import cv2

vs = cv2.VideoCapture(0) # parameter must be the array index of the camera of ur pc u are using .. trial and error to find(0->5 max)

while True : # infinite loop for showing imshow() continuously to make it look like video !! 
    _, img = vs.read() # read() returns two parameters where the 1st is not necessary so we have put '_' to ignore it
    cv2.imshow("Video Stream", img)
    key = cv2.waitKey(1) & 0xFF  ## Very complex one .. .. 
    # here waitKey return the unicode of the key pressed, after adding it with hexadecimal value which is all 8 high bits 
    # implying to take only 8 bit information
    
    if key == ord('q') : # converts the character 'q' to the unicode value
        break
vs.release()
cv2.destroyAllWindows()