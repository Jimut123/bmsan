#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 01/12/2020 15:56
# @Author  : Jimut Bahan Pal

import glob
import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam, Nadam
from tensorflow.keras.applications import MobileNetV2
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
#tf.random.set_seed(221)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

import sys
sys.path.insert(0, '../../')
from models import DRRMSAN_multiscale_attention_bayes_001

img_files = glob.glob('../trainx/*.bmp')
msk_files = glob.glob('../trainy/*.bmp')

img_files.sort()
msk_files.sort()
print(img_files[:10])
print(msk_files[:10])
print(len(img_files))
print(len(msk_files))


X = []
Y = []

for img_fl in tqdm(img_files):
    if(img_fl.split('.')[-1]=='bmp'):
        img = cv2.imread('{}'.format(img_fl), cv2.IMREAD_COLOR)
        X.append(img) #resized_img)
        img_msk = "../trainy/Y_img_"+str(img_fl.split('.')[2]).split('_')[-1]+".bmp"
        msk = cv2.imread('{}'.format(img_msk), cv2.IMREAD_GRAYSCALE)
        Y.append(msk)#resized_msk)


print(len(X))
print(len(Y))

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(Y_train.shape)

Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))


Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))


X_train = X_train / 255
X_test = X_test / 255
Y_train = Y_train / 255
Y_test = Y_test / 255

Y_train = np.round(Y_train,0)
Y_test = np.round(Y_test,0)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)








def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union

def saveModel(model):

    model_json = model.to_json()

    try:
        os.makedirs('models')
    except:
        pass

    fp = open('models/modelP_attn_1_bothsides_skinleison.json','w')
    fp.write(model_json)
    model.save_weights('models/modelW_attn_1_bothsides_skinleison.h5')


jaccard_index_list = []
dice_coeff_list = []

def evaluateModel(model, X_test, Y_test, batchSize):

    try:
        os.makedirs('results')
    except:
        pass


    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)

    yp = np.round(yp,0)
    yp = yp[4]

    for i in range(10):

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
        plt.title('Prediction')

        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jacard = (np.sum(intersection)/np.sum(union))
        plt.suptitle('Jacard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard))

        plt.savefig('results/'+str(i)+'.png',format='png')
        plt.close()


    jacard = 0
    dice = 0


    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()

        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection)/np.sum(union))

        dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))


    jacard /= len(Y_test)
    dice /= len(Y_test)



    print('Jacard Index : '+str(jacard))
    print('Dice Coefficient : '+str(dice))
    with open("Output.txt", "w") as text_file:
        text_file.write("Jacard : {} Dice Coef : {} ".format(str(jacard), str(dice)))

    jaccard_index_list.append(jacard)
    dice_coeff_list.append(dice)
    fp = open('models/log_attn_1_bothsides_skinleison.txt','a')
    fp.write(str(jacard)+'\n')
    fp.close()

    fp = open('models/best_attn_1_bothsides_skinleison.txt','r')
    best = fp.read()
    fp.close()

    if(jacard>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard))
        print('***********************************************')
        fp = open('models/best_attn_1_bothsides_skinleison.txt','w')
        fp.write(str(jacard))
        fp.close()

        saveModel(model)



def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):
    #epochs = 10
    #for epoch in range(epochs):
    #print('Epoch : {}'.format(epoch+1))
    history = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=epochs, verbose=1)

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)



    # save to json:
    hist_json_file = 'history_attn_1_bothsides_skinleison.json'
    # with open(hist_json_file, 'a') as out:
    #     out.write(hist_df.to_json())
    #     out.write(",")
    #     out.close()

    with open(hist_json_file, mode='w') as f:
       hist_df.to_json(f)

    # or save to csv:
    hist_csv_file = 'history_attn_1_bothsides_skinleison.csv'
    # with open(hist_csv_file, 'a') as out:
    #     out.write(str(hist_df.to_csv()))
    #     out.write(",")
    #     out.close()


    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    evaluateModel(model,X_test, Y_test,batchSize)

    return model

alpha_1 = 0.25
alpha_2 = 0.25
alpha_3 = 0.25
alpha_4 = 0.25
model = DRRMSAN_multiscale_attention_bayes_001(height=192, width=256, n_channels=3, alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_3 = alpha_3, alpha_4 = alpha_4)

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])
from tensorflow.keras.metrics import Recall, Precision


# 4.55e-6
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=[dice_coef, jacard, Recall(), Precision() ,'accuracy'])

saveModel(model)

fp = open('models/log.txt','w')
fp.close()
fp = open('models/best.txt','w')
fp.write('-1.0')
fp.close()

trainStep(model, X_train, Y_train, X_test, Y_test, epochs=20, batchSize=2)


