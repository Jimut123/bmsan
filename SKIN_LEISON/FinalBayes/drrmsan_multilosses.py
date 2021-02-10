import os
import numpy as np
import cv2
from glob import glob
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
from models import DRRMSAN_multiscale_attention_bayes

def get_dice_from_alphas(alpha_1, alpha_2, alpha_3, alpha_4):
    PATH = ""
    np.random.seed(42)
    tf.random.set_seed(42)

    ## Hyperparameters

    #IMG_SIZE = 256
    EPOCHS = 10
    BATCH = 2
    LR = 1e-5

    def load_data(path, split=0.2):
        """
        from glob import glob
        images_list = sorted(glob(os.path.join(path, "trainx/*.bmp")))
        masks_list = sorted(glob(os.path.join(path, "trainy/*.bmp")))
        """

        import sys
        import glob
        from tqdm import tqdm
        #insert :: sys.path.insert(0, '../../')
        
        ############################## insert:: add ../ before two
        img_files = glob.glob('trainx/*.bmp')
        msk_files = glob.glob('trainy/*.bmp')

        images_list = []
        masks_list = []


        for img_fl in tqdm(img_files):
            if(img_fl.split('.')[-1]=='bmp'):
            images_list.append(img_fl)
            # insert :: img_msk = "../trainy/Y_img_"+str(img_fl.split('.')[2]).split('_')[-1]+".bmp"
            img_msk = "trainy/Y_img_"+str(img_fl.split('.')[0]).split('_')[-1]+".bmp"
            #print("----",img_msk)
            #break
            masks_list.append(img_msk)
        
        
        tot_size = len(images_list)
        test_size = int(split * tot_size)
        val_size = int(split * (tot_size - test_size))

        x_train, x_val = train_test_split(images_list, test_size=val_size, random_state=42)
        y_train, y_val = train_test_split(masks_list, test_size=val_size, random_state=42)

        x_train, x_test = train_test_split(x_train, test_size=test_size, random_state=42)
        y_train, y_test = train_test_split(y_train, test_size=test_size, random_state=42)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


    def read_img(path):
        path = path.decode()
        tmp = cv2.imread(path, cv2.IMREAD_COLOR)
        tmp = cv2.resize(tmp, (256, 192))
        tmp = tmp/255.0
        return tmp

    def read_mask(path):
        path = path.decode()
        tmp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        tmp = cv2.resize(tmp, (256, 192))
        tmp = tmp/255.0
        tmp = np.expand_dims(tmp, axis=-1)
        return tmp

    def tf_parse(a, b):
        def _parse(a, b):
            a = read_img(a)
            b = read_mask(b)
            return a, b

        a, b = tf.numpy_function(_parse, [a, b], [tf.float64, tf.float64])
        a.set_shape([192, 256, 3])
        b.set_shape([192, 256, 1])
        return a, b

    def tf_dataset(a, b, batch=32):
        data = tf.data.Dataset.from_tensor_slices((a, b))
        data = data.map(tf_parse)
        data = data.batch(batch)
        data = data.repeat()
        return data

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(PATH)

    print("Training data: ", len(x_train))
    print("Validation data: ", len(x_val))
    print("Testing data: ", len(x_test))


    def read_and_rgb(a):
        a = cv2.imread(a)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        return a



    model = DRRMSAN_multiscale_attention_bayes(height=192, width=256, n_channels=3, alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_3 = alpha_3, alpha_4 = alpha_4)
    model.summary()


    smooth = 1e-15
    def dice_coef(y_true, y_pred):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    def dice_loss(y_true, y_pred):
        return 1.0 - dice_coef(y_true, y_pred)

    def jacard(y_true, y_pred):

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum ( y_true_f * y_pred_f)
        union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

        return intersection/union


    train_data = tf_dataset(x_train, y_train, batch=BATCH)
    valid_data = tf_dataset(x_val, y_val, batch=BATCH)

    opt = tf.keras.optimizers.Nadam(LR)
    metrics = [dice_coef, jacard, Recall(), Precision() ,'accuracy']
    model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)

    from datetime import datetime
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow import keras

    # for storing logs into tensorboard
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")


    callbacks = [
        #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        #EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
        ModelCheckpoint("./model_checkpoint", monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir=logdir)
    ]

    train_steps = len(x_train)//BATCH
    valid_steps = len(x_val)//BATCH

    if len(x_train) % BATCH != 0:
        train_steps += 1
    if len(x_val) % BATCH != 0:
        valid_steps += 1


    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )

    import pandas as pd
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)



    # save to json:
    hist_json_file = 'history_skin_drrmsan.json'
    # with open(hist_json_file, 'a') as out:
    #     out.write(hist_df.to_json())
    #     out.write(",")
    #     out.close()

    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv:
    hist_csv_file = 'history_skin_drrmsan.csv'
    # with open(hist_csv_file, 'a') as out:
    #     out.write(str(hist_df.to_csv()))
    #     out.write(",")
    #     out.close()


    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


    model.save_weights("skin_drrmsan_150e.h5")
    model.save("skin_drrmsan_with_weight_150e.h5")

    # Run this module only while loading the pre-trained model.
    model = load_model('skin_drrmsan_with_weight_150e.h5',custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef, 'jacard':jacard})
    model.summary()



    jaccard_index_list = []
    dice_coeff_list = []

    def evaluateModel(model, X_test, Y_test, batchSize):  
        
        try:
            os.makedirs('results')
        except:
            pass


        yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
        

        yp = np.round(yp,0)
        # get the actual 4rth output
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
        fp = open('models/log_drrmsan_skinleison.txt','a')
        fp.write(str(jacard)+'\n')
        fp.close()

        fp = open('models/best_drrmsan_skinleison.txt','r')
        best = fp.read()
        fp.close()

        if(jacard>float(best)):
            print('***********************************************')
            print('Jacard Index improved from '+str(best)+' to '+str(jacard))
            print('***********************************************')
            fp = open('models/best_UNet_skinleison.txt','w')
            fp.write(str(jacard))
            fp.close()

            #saveModel(model)


    from tqdm import tqdm

    X_test = []
    Y_test = []

    for img_fl, img_msk in tqdm(zip(x_test, y_test)):
        img = cv2.imread('{}'.format(img_fl), cv2.IMREAD_COLOR)
        X_test.append(img)
        #img_msk = "../trainy/Y_img_"+str(img_fl.split('.')[2]).split('_')[-1]+".bmp"
        msk = cv2.imread('{}'.format(img_msk), cv2.IMREAD_GRAYSCALE)
        Y_test.append(msk)#resized_msk)



    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))

    X_test = X_test / 255
    Y_test = Y_test / 255


    #Y_test = np.round(Y_test,0)	

    print(X_test.shape)
    print(Y_test.shape)


    try:
        os.makedirs('models')
    except:
        pass

    fp = open('models/log_drrmsan_skinleison.txt','w')
    fp.close()
    fp = open('models/best_drrmsan_skinleison.txt','w')
    fp.write('-1.0')
    fp.close()

    evaluateModel(model, X_test, Y_test, BATCH)


    import json
    import matplotlib.pyplot as plt

    with open('history_skin_drrmsan.json', 'r') as f:
        array = json.load(f)
    #print (array)
    #print(json.dumps(array, indent=4, sort_keys=True))

    for item in array:
        if 'val' in item and 'dice' in item and 'add' in item: # and 'activation' in item:#
            val = array[item]['19']
            print("Dice Value got = ",val)
    return val


