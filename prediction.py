import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dataset import fetch_openml
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml("mnist_784",version = 1,return_X_y = True)
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,train_size = 7500,test_size =2500,random_state = 9)

Xtrain = Xtrain/255.0
Xtrain = Xtrain/255.0

lr = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(Xtrain,Ytrain)

def get_prediction(image):
    img_pil = Image.open(image)
    img_bw = img_pil.convert("L")
    img_bw_resized = img_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resized,pixel_filter)
    img_bw_resized_scaled = np.clip(img_bw_resized-min_pixel,0,255)
    max_pixel = np.max(img_bw_resized)
    img_bw_resized_scaled = np.asarray(img_bw_resized_scaled)/max_pixel
    test_sample = np.array(img_bw_resized_scaled).reshape(1,784)
    test_predict = lr.predict(test_sample)
    return test_predict[0]
    