# libraries used here opencv, PIL(satelitte image processing),scikit-image, mahata, sciPy, and numpy 

# display an image even with diff extensions of images 
import cv2 # importing opencv library
img = cv2.imread("vision_1.png")  # reading an image directly
cv2.imwrite("new_vision_1.jpg", img) # saving th read image in another image file
cv2.imshow("Newvision logo", img) # showing the image with the "Newvision logo" name.
#image properties
print(img.size)
print(img.shape)
print(img.dtype) # and many more such so
cv2.waitKey(0) # // values is the amount of time it waits to close the image.. putting it 0 means it waits infinitely for any key to be pressed.. any +ve x means x ms time.
cv2.destroyAllWindows() # destroy all windows oh high GUI windows


# # Conver the image from color to gray scale image.
# # U CAN PLAY BY CHANGING IT TO rgb2bgr which is cool .. and bunch of other conversions 
# import cv2 
# img = cv2.imread("vision_1.png") 
# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image function and COLOR_BGR2GRAY methos to convert rgb img into gray scale image
# blackWhiteImg = cv2.threshold(grayImg, 220, 255, cv2.THRESH_BINARY)[1] # inorder to produce onlu black and white image, so the threshold function places all below 128 as 0 and above till max as white so ony  B/W
# cv2.imwrite("grayImageFromRGB.png", grayImg)
# cv2.imshow("Original color(RGB) Image",img)
# cv2.imshow("Gray image of original image",grayImg)
# cv2.imshow("Just Black and White image", blackWhiteImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



