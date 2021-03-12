#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 01/12/2020 15:56
# @Author  : Jimut Bahan Pal

import glob
import json
import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, recall_score

import sys
sys.path.insert(0, '../../')
from models import  unet


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

kf = KFold(n_splits=5)
kf.get_n_splits(X)

fold_no = 0
for train_index, test_index in kf.split(X):
    fold_no += 1
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
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


    import numpy as np
    """
    >>> x = np.zeros((100, 12, 12, 3))
    >>> x.shape
    (100, 12, 12, 3)
    >>> y = np.transpose(x)
    >>> y.shape
    (3, 12, 12, 100)
    >>> z = np.moveaxis(y,-1,0)
    >>> z.shape
    (100, 3, 12, 12)

    """
    X_train = np.moveaxis(X_train,-1,1)
    print(X_train.shape)

    Y_train = np.moveaxis(Y_train,-1,1)
    Y_train = np.repeat(Y_train,repeats=3,axis=1)
    print(Y_train.shape)

    X_test = np.moveaxis(X_test,-1,1)
    print(X_test.shape)

    Y_test = np.moveaxis(Y_test,-1,1)
    Y_test = np.repeat(Y_test,repeats=3,axis=1)
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

        fp = open('models/modelP_orunet_skinLesion.json','w')
        fp.write(model_json)
        model.save_weights('models/modelW_orunet_skinLesion.h5')


    jaccard_index_list = []
    dice_coeff_list = []

    def evaluateModel(model, X_test, Y_test, batchSize):

        try:
            os.makedirs('results_{}'.format(fold_no))
        except:
            pass


        yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)

        yp = np.round(yp,0)

        for i in range(10):
            plt.figure(figsize=(20,10))
            plt.subplot(1,3,1)
            plt.imshow(np.moveaxis(X_test[i],0,-1))
            plt.title('Input')
            plt.subplot(1,3,2)
            plt.imshow(np.moveaxis(Y_test[i],0,-1))
            plt.title('Ground Truth')
            plt.subplot(1,3,3)
            plt.imshow(np.moveaxis(yp[i],0,-1))
            plt.title('Prediction')

            intersection = yp[i].ravel() * Y_test[i].ravel()
            union = yp[i].ravel() + Y_test[i].ravel() - intersection

            avg_precision = average_precision_score(yp[i].ravel(), Y_test[i].ravel())
            dice = (2. * np.sum(intersection)) / (np.sum(yp[i].ravel()) + np.sum(Y_test[i].ravel()))

            jacard = (np.sum(intersection)/np.sum(union))
            plt.suptitle('Jacard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard)
            +" Dice : "+str(dice)+ " Precision : "+str(avg_precision))

            plt.savefig('results_{}/'.format(fold_no)+str(i)+'.png',format='png')
            plt.close()
        
        jacard = 0
        dice = 0
        avg_precision = 0
        recall_score = 0

        for i in range(len(Y_test)):
            yp_2 = yp[i].ravel()
            y2 = Y_test[i].ravel()

            intersection = yp_2 * y2
            union = yp_2 + y2 - intersection
            avg_precision += average_precision_score(yp_2, y2)
            # recall_score += recall_score(yp_2, y2)

            jacard += (np.sum(intersection)/np.sum(union))

            dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))


        jacard /= len(Y_test)
        dice /= len(Y_test)
        avg_precision /= len(Y_test)
        # recall_score /= len(Y_test)

        print('Jacard Index : '+str(jacard))
        print('Dice Coefficient : '+str(dice))
        with open("Output.txt", "a") as text_file:
            text_file.write("Fold = {} Jacard : {} Dice Coef : {} Avg. Precision : {}  \n".format(str(fold_no), 
            str(jacard), str(dice), str(avg_precision)))
        

        jaccard_index_list.append(jacard)
        dice_coeff_list.append(dice)
        fp = open('models/log_orunet_skinLesion.txt','a')
        fp.write(str(jacard)+'\n')
        fp.close()

        fp = open('models/best_orunet_skinLesion.txt','r')
        best = fp.read()
        fp.close()

        if(jacard>float(best)):
            print('***********************************************')
            print('Jacard Index improved from '+str(best)+' to '+str(jacard))
            print('***********************************************')
            fp = open('models/best_orunet_skinLesion.txt','w')
            fp.write(str(jacard))
            fp.close()

            saveModel(model)

    def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):

        history = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=epochs, verbose=1)

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)



        # save to json:
        hist_json_file = 'history_orunet_skinLesion_fold_{}.json'.format(fold_no)
        # with open(hist_json_file, 'a') as out:
        #     out.write(hist_df.to_json())
        #     out.write(",")
        #     out.close()

        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)

        # or save to csv:
        hist_csv_file = 'history_orunet_skinLesion_fold_{}.csv'.format(fold_no)
        # with open(hist_csv_file, 'a') as out:
        #     out.write(str(hist_df.to_csv()))
        #     out.write(",")
        #     out.close()


        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        evaluateModel(model,X_test, Y_test,batchSize)

        return model
    # img_w, img_h, n_label, data_format='channels_first'
    model = unet(img_h=256, img_w=192, n_label=3)

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])
    model.compile(optimizer=Adam(learning_rate=1e-5),loss='binary_crossentropy',metrics=[dice_coef, jacard, Recall(), Precision(), 'accuracy'])

    saveModel(model)

    fp = open('models/log_orunet_skinLesion.txt','w')
    fp.close()
    fp = open('models/best_orunet_skinLesion.txt','w')
    fp.write('-1.0')
    fp.close()

    trainStep(model, X_train, Y_train, X_test, Y_test, epochs=150, batchSize=2)

