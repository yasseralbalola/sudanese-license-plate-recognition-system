import cv2
import numpy as np 


#represent the Support Vector Machine model
class SVM():
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

    def save(self, fn):
        self.model.save(fn)

    def load(self,name):
        return cv2.ml.SVM_load(name)

#initialize the Histogram of Oriented Gradient descripter
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

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
    	cellSize, nbins, derivAperture, winSigma, histogramNormType, 
        L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    return hog

#train a model to recognise both arabic and english numbers, by using SVM as traing model, and HOG as feature descripter
def train(lan):
    path = ''

    if lan == 'ar':
        path = 'numbers/arabic/'
    elif lan == 'en':
        path = 'numbers/english/'
    else:
        #add exception handling
        return

    hog = get_hog()
    labels=[i for i in range(10)]
    labels = np.array(labels)

    hogs=[]

    for i in range(10):
        img = cv2.imread(path + str(i) + ".jpg")
        labels += [i]
        
        hogs += [hog.compute(img)]


    hogs = np.array(hogs)

    model = SVM()
    model.train(hogs,labels)
    model.save(lan + "_digits_trained.dat")

    return model

#load a trained model
def load(name):
    return cv2.ml.SVM_load(name)
