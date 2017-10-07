import cv2
import numpy as np 


class SVM():
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

    def save(self, fn):
        self.model.save(fn)

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



def rec(components,lan):
    result = []
    hog = get_hog()
    model = cv2.ml.SVM_load("ar_digits_svm.dat")

    for com in components:

		result += [int(model.predict(np.array([hog.compute(com)]))[1].ravel()[0])]

    return result



def recognize(ar_com,en_com):

	result = []

	ar_result = rec(ar_com,'ar')
	en_result = rec(en_com,'en')

	result = ar_result


	return result


