import cv2
import numpy as np 

def load():    
    ar_model = cv2.ml.SVM_load("ar_digits_trained.dat")
    return ar_model

def predict(model, image):
    prediction = int(model.predict(image)[1].ravel()[0])
    return prediction

def recognize(components):
    model = load()
    predictions = []

    for i in components:
        predictions += [predict(model,i)]
    
    return predictions
