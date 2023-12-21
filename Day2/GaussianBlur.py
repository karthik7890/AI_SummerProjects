import cv2
import imutils


img = cv2.imread("vision_1.png")
# # Application of imutils 
# resizedImg = imutils.resize(img, width=100) # could be width, height - they are same !
# cv2.imwrite("resizedImg.png", resizedImg) # image changes with the change in the parmaters in the above line.. 
# cv2.imshow("Resized Image", resizedImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# ----------------------
#

# SMOTHEING OF IMAGE - BLUR IS OBSERVED(actually a side-product - if done on a image actually needs a filter)... We used here Gaussian filter and blur functionality
# For a normal image, the filters use is not seen !! 

#syntax :  dest = cv2.GaussianBlur(src, kernel, border Type) .. here kernel is the matrix size of the filter applied to the img.. 
# .. so there could be restrictions of how big u can use as u can't exceed the size of the img itself !!  
gaussianBlurImg = cv2.GaussianBlur(img, (25,25), 0) 
cv2.imwrite("GaussianBlurImage.png", gaussianBlurImg)
cv2.imshow("Gaussian Blur Image", gaussianBlurImg) # u can read the docs for this func for more notes and for explaining parameters or if using any intelligence/vscode..
# ...it will give a brief info on function u put cursor on -- very very USEFUL !!! 

cv2.waitKey(0) # Make sure u use this whenever imshow() is used, as it will not show the img otherwise as it waits no time to display the img.. 
cv2.destroyAllWindows() # mostly not necessary 

