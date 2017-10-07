import cv2
import numpy as np 

def preprocessing(img,lan):
	
	if lan == "ar":
		img=img[66:185,200:]
	elif lan == "en":
		img=img[186:240,200:]
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

def seg(plate,lan):

	rows_x = 0

	if lan == "ar":
		rows_x = 119

	elif lan == "en":
		rows_x = 54

	img = cv2.resize(plate,(500,246))
	img=preprocessing(img,lan)

	kernel=np.ones((3,3))
	black=np.zeros((rows_x,300),np.uint8)
	components=[]

	while True:
		white_point=get_white_pixel(img)

		blank_img=np.zeros((rows_x,300),np.uint8)

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
		
		if c_row > 10 and c_col > 10:
		

			component = cv2.resize(component,(60,60))
			component = np.array(component)
			components += [component]
		
		img=image_difference(img,blank_img)

		if is_same_image(img,black):
			break


	# components = np.array(components)
	return components


def segment(plate):

	ar_com = seg(plate,'ar')
	en_com = seg(plate,'en')

	return ar_com,en_com

# if __name__ == '__main__':
# 	img=cv2.imread('8.jpg',0)
# 	com=segment(img,'en')
# 	for c in com:
# 		cv2.imshow('image',c)
# 		cv2.waitKey(0)
# 		cv2.destroyAllWindows()
