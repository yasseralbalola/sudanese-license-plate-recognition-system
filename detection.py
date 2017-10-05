import cv2
import numpy as np
from operator import itemgetter



np.set_printoptions(threshold=np.nan)
img = cv2.imread("image.jpg",0)
ref = cv2.imread("lp.jpg",0)
summation=0
for i in range(246):
	for j in range(500):
		summation+=ref[i][j]

size=ref.shape[0]*ref.shape[1]

mean=summation/size
ref=ref.astype(np.int16)
ref=ref-mean
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
noise_removal = cv2.bilateralFilter(img,9,75,75)
equal_histogram = cv2.equalizeHist(noise_removal)
ret,thresh_image = cv2.threshold(equal_histogram,0,255,cv2.THRESH_OTSU)
canny_image = cv2.Canny(thresh_image,250,255)
new,contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
license_plate_candidates = []
for c in contours:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.06 * peri, True)
	area=cv2.contourArea(approx)

	if len(approx) == 4 and area > 2000:  
		license_plate_candidates += [approx]

results=[]

for license_plate_candidate in license_plate_candidates:
	
	r1=license_plate_candidate[0][0][0]
	c1=license_plate_candidate[0][0][1]
	r2=license_plate_candidate[1][0][0]
	c2=license_plate_candidate[1][0][1]
	r3=license_plate_candidate[2][0][0]
	c3=license_plate_candidate[2][0][1]
	r4=license_plate_candidate[3][0][0]
	c4=license_plate_candidate[3][0][1]

	license_plate_candidate=[[r1,c1],[r2,c2],[r3,c3],[r4,c4]]
	license_plate_candidate=sorted(license_plate_candidate,key=itemgetter(1))
	up=sorted(license_plate_candidate[:2],key=itemgetter(0))
	down=sorted(license_plate_candidate[2:],key=itemgetter(0))

	license_plate_candidate=[up[0],down[0],up[1],down[1]]

	r1=license_plate_candidate[0][1]
	c1=license_plate_candidate[0][0]
	r2=license_plate_candidate[1][1]
	c2=license_plate_candidate[1][0]
	r3=license_plate_candidate[2][1]
	c3=license_plate_candidate[2][0]
	r4=license_plate_candidate[3][1]
	c4=license_plate_candidate[3][0]



	pts1 = np.float32([[c1,r1],[c2,r2],[c3,r3],[c4,r4]])
	pts2 = np.float32([[0,0],[0,246],[500,0],[500,246]])


	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img,M,(500,246))
	summation=0
	for i in range(246):
		for j in range(500):
			summation+=dst[i][j]

	size=dst.shape[0]*dst.shape[1]

	mean=summation/size
	dst=dst.astype(np.int16)
	dst=dst-mean

	correlation=0
	for i in range(246):
		for j in range(500):
			correlation+=(dst[i][j]*ref[i][j])
	

	results+=[correlation]


license_plate=license_plate_candidates[results.index(max(results))]

r1=license_plate[0][0][0]
c1=license_plate[0][0][1]
r2=license_plate[1][0][0]
c2=license_plate[1][0][1]
r3=license_plate[2][0][0]
c3=license_plate[2][0][1]
r4=license_plate[3][0][0]
c4=license_plate[3][0][1]
license_plate=[[r1,c1],[r2,c2],[r3,c3],[r4,c4]]
license_plate=sorted(license_plate,key=itemgetter(1))
up=sorted(license_plate[:2],key=itemgetter(0))
down=sorted(license_plate[2:],key=itemgetter(0))
license_plate=[up[0],down[0],up[1],down[1]]
r1=license_plate[0][1]
c1=license_plate[0][0]
r2=license_plate[1][1]
c2=license_plate[1][0]
r3=license_plate[2][1]
c3=license_plate[2][0]
r4=license_plate[3][1]
c4=license_plate[3][0]

pts1 = np.float32([[c1,r1],[c2,r2],[c3,r3],[c4,r4]])
pts2 = np.float32([[0,0],[0,246],[500,0],[500,246]])


M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(500,246))










	# for i in range(246):
	# 	for j in range(500):




# cv2.imshow('img',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("lp.jpg",dst)