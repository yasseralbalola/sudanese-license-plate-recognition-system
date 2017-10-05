import cv2
import numpy as np 


class StatModel(object):
    # def load(self, fn):
    #     self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def load(self,name):
        return cv2.ml.SVM_load(name)

    def predict(self, samples):

        return self.model.predict(samples)[1].ravel()

def get_hog() : 
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog

hog = get_hog()

labels=[]
hogs=[]

for i in range(10):
	img = cv2.imread('Untitled Folder 2/'+  str(i)+".jpg")
	labels += [i]
	
	hogs += [hog.compute(img)]


hogs = np.array(hogs)
labels = np.array(labels)

model = cv2.ml.SVM_load("digits_svm.dat")


# model.train(hogs, labels)

# model.save('digits_svm.dat')



test = cv2.imread('Untitled Folder 2/4.jpg')

print int(model.predict(np.array([hog.compute(test)]))[1].ravel()[0])
