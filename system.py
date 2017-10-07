import cv2
import numpy as np 
import detection
import segmentation
import recognition

img = cv2.imread('image.jpg',0)

cv2.imshow('plate',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plate = detection.detect(img)

cv2.imshow('plate',plate)
cv2.waitKey(0)
cv2.destroyAllWindows()


ar_com,en_com = segmentation.segment(plate)

for i in ar_com:

	cv2.imshow('plate',i)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

for i in en_com:

	cv2.imshow('plate',i)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

result = recognition.recognize(ar_com,en_com)


print result