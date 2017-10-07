import cv2
import numpy as np 

def preprocessing(img):
	img=img[66:185,200:]
	img=cv2.bitwise_not(img)
	# max_pixel=max([max(i) for i in img])
	# min_pixel=min([min(i) for i in img])
	# avg_pixel= (int(max_pixel)+int(min_pixel))/2
	threshold_value,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel=np.ones((3,3))

	img = cv2.erode(img,kernel,iterations = 1)
	img = cv2.dilate(img,kernel,iterations = 2)
	img = cv2.erode(img,kernel,iterations = 1)

	return img

def get_white_pixel(img):
	row,col=img.shape
	r=0
	c=0
	for i in range(col):
		if sum(img[:,i]) != 0:
			c=i
			break
	for i in range(row):
		if img[i,c] != 0:
			r=i
			break


	# for i in range(col):
	# 	for j in range(row):
	# 		if img[j,i] !=0:
	# 			white_point=[j,i]
	# 			end_loop=True
	# 			break
	# 	if end_loop:
	# 		break
	return [r,c]

def is_same_image(img1,img2):
	return np.all(img1 == img2)

def image_difference(img1,img2):
	img2=cv2.bitwise_not(img2)
	out_img=cv2.bitwise_and(img1,img1,mask=img2)
	return out_img

def segment(plate):
	img = cv2.resize(plate,(500,246))
	img=preprocessing(img)

	kernel=np.ones((3,3))
	black=np.zeros((119,300),np.uint8)
	components=[]

	while True:
		white_point=get_white_pixel(img)
		blank_img=np.zeros((119,300),np.uint8)

		_,blank_img = cv2.threshold(blank_img,125,255,cv2.THRESH_BINARY)
			
		blank_img[white_point[0],white_point[1]]=255
		
		stop_of_dilation=blank_img

		while True:
			blank_img = cv2.dilate(blank_img,kernel,iterations = 1)
			blank_img=cv2.bitwise_and(blank_img,blank_img,mask=img)
			if is_same_image(blank_img,stop_of_dilation) == False:
				stop_of_dilation=blank_img
			else:
				break

		im, contours, hierarchy = cv2.findContours(blank_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		contour=contours[0]
		x,y,w,h = cv2.boundingRect(contour)
		component=blank_img[y:y+h,x:x+w]
		
		#charachters constrains goes here
		c_row,c_col=component.shape
		
		# if c_row < 100 and c_row > 20 and c_col > 15 and c_col < 60:
		

		component = cv2.resize(component,(60,60))
		component = np.array(component)
		components += [component]
		
		img=image_difference(img,blank_img)

		if is_same_image(img,black):
			break

	return components