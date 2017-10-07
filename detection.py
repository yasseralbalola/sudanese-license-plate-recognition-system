import cv2
import numpy as np
from operator import itemgetter

def normalize(image):
	summation = sum(map(sum,image))
	size  = image.shape[0]*image.shape[1]
	mean  = summation/size
	image = image.astype(np.int16)
	image = image-mean
	return image

def get_contours(image):
	noise_removal = cv2.bilateralFilter(image,9,75,75)
	equal_histogram = cv2.equalizeHist(noise_removal)
	ret,thresh_image = cv2.threshold(equal_histogram,0,255,cv2.THRESH_OTSU)
	canny_image = cv2.Canny(thresh_image,250,255)
	new,contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours

def get_candidates(contours):
	candidates = []

	for contour in contours:
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
		if len(approx) != 4: continue
		
		area=cv2.contourArea(approx)
		if area > 2000:  
			candidates += [approx]

	return candidates

def extract(candidates):
	r1=candidate[0][0][0]
	c1=candidate[0][0][1]
	r2=candidate[1][0][0]
	c2=candidate[1][0][1]
	r3=candidate[2][0][0]
	c3=candidate[2][0][1]
	r4=candidate[3][0][0]
	c4=candidate[3][0][1]

	candidate=[[r1,c1],[r2,c2],[r3,c3],[r4,c4]]
	candidate=sorted(candidate,key=itemgetter(1))
	up=sorted(candidate[:2],key=itemgetter(0))
	down=sorted(candidate[2:],key=itemgetter(0))

	candidate=[up[0],down[0],up[1],down[1]]

	r1=candidate[0][1]
	c1=candidate[0][0]
	r2=candidate[1][1]
	c2=candidate[1][0]
	r3=candidate[2][1]
	c3=candidate[2][0]
	r4=candidate[3][1]
	c4=candidate[3][0]

	pts1 = np.float32([[c1,r1],[c2,r2],[c3,r3],[c4,r4]])
	pts2 = np.float32([[0,0],[0,246],[500,0],[500,246]])

	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img,M,(500,246))

	return dst

def get_correlation(dst,ref):
	correlation=0
	for i in range(246):
		for j in range(500):
			correlation+=(dst[i][j]*ref[i][j])

	return correlation

def detect(image):
	np.set_printoptions(threshold=np.nan)
	img = image
	ref = cv2.imread("../plate.jpeg",0)
	
	ref = normalize(ref)
	contours = get_contours(img)
	license_plate_candidates = get_candidates(contours)

	results=[]

	for license_plate_candidate in license_plate_candidates:
		dst = extract(license_plate_candidate)		
		dst = normalize(dst)
		results += [get_correlation(dst,ref)]

	license_plate = license_plate_candidates[results.index(max(results))]

	plate = extract(license_plate)

	return plate