#Import Modules

#GPyOpt - Cases are important, for some reason
import GPyOpt
from GPyOpt.methods import BayesianOptimization

#numpy
import numpy as np
from numpy.random import multivariate_normal #For later example

import pandas as pd

#Plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.random import multivariate_normal

# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
#%pylab inline
import GPyOpt
import GPy
import numpy as np
import pickle

# import drrmsan_multilosses

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision 
from sklearn.metrics import average_precision_score, recall_score
from tensorflow.keras import backend as K
import sys
sys.path.insert(0, '../../')
from models import DRRMSAN_multiscale_attention_bayes_022_attn_3


from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)
#session_config = tf.ConfigProto()
#session_config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, session_config=session_config)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

PATH = ""
np.random.seed(42)
tf.random.set_seed(42)
## Hyperparameters

#IMG_SIZE = 256
EPOCHS = 1
BATCH = 2
LR = 1e-5

alpha_1 = 0.0; alpha_2 = 0.0; alpha_3 = 0.0; alpha_4 = 0.0;

########################################################

def load_data(path, split=0.2):

    import sys
    import glob
    from tqdm import tqdm
    #insert :: sys.path.insert(0, '../../')
    
    ############################## insert:: add ../ before two

    images_list = []
    masks_list = []

    img_files = sorted(glob.glob('../ISIC-2017_Training_Data/ISIC_*.jpg'))
    msk_files = sorted(glob.glob('../ISIC-2017_Training_Data/*_superpixels.png'))


    img_files.sort()
    msk_files.sort()

    print("B==>",len(img_files))
    print(len(msk_files))


    X = []
    Y = []

    for img_fl in tqdm(img_files):
        images_list.append(img_fl)
        im_name = str(str(img_fl.split('.')[2]).split('/')[2]).split('_')[1]
        mask_name = '../ISIC-2017_Training_Data/ISIC_'+im_name+'_superpixels.png'
        masks_list.append(mask_name)
    
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
    tmp = cv2.resize(tmp, (256, 256))
    tmp = tmp/255.0
    return tmp

def read_mask(path):
    path = path.decode()
    tmp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    tmp = cv2.resize(tmp, (256, 256))
    tmp = tmp/255.0
    tmp = np.expand_dims(tmp, axis=-1)
    return tmp

def tf_parse(a, b):
    def _parse(a, b):
        a = read_img(a)
        b = read_mask(b)
        return a, b

    a, b = tf.numpy_function(_parse, [a, b], [tf.float64, tf.float64])
    a.set_shape([256, 256, 3])
    b.set_shape([256, 256, 1])
    return a, b

def tf_dataset(a, b, batch=32):
    data = tf.data.Dataset.from_tensor_slices((a, b))
    data = data.map(tf_parse)
    data = data.batch(batch)
    data = data.repeat()
    return data


def read_and_rgb(a):
    a = cv2.imread(a)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    return a


smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def jaccard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union


def evaluateModel(model, X_test, Y_test, batchSize):
    global alpha_1, alpha_2, alpha_3, alpha_4

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
    # global dice 
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
    
    fp = open('models/log_drrmsan_isic.txt','a')
    fp.write(str(jacard)+'\n')
    fp.close()

    fp = open('models/best_drrmsan_isic.txt','r')
    best = fp.read()
    fp.close()

    if(jacard>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard))
        print('***********************************************')
        fp = open('models/best_UNet_isic.txt','w')
        fp.write(str(jacard))
        fp.close()

        #saveModel(model)
    
    print("00"*50)
    f = open("./_bayesian_opt_logs.txt", "a+")
    dump_str = str(alpha_1) + " " + str(alpha_2) + " " + str(alpha_3) + " " + str(alpha_4) + " " + str(dice) + " " + str(jacard) + " " + str(dice*jacard) +" \n"
    f.write(dump_str)
    f.close()
    print("Dice Value Used = ", -float(dice))
    print("Jacard Value Used = ", -float(jacard))
    # del model
    print("Model deleted and dice value returned!!")
    return dice, jacard




def f(x):
    # x is a 4D vector.
    # Function which will send alpha_1, alpha_2, alpha_3 and alpha_4
    # to the actual model and will get the dice coefficient in return.
    global alpha_1, alpha_2, alpha_3, alpha_4
    
    alpha_1 = x[:, 0][0]
    alpha_2 = x[:, 1][0]
    alpha_3 = x[:, 2][0]
    alpha_4 = x[:, 3][0]
    print(alpha_1, " ", alpha_2," ",alpha_3," ",alpha_4)

    # dice = drrmsan_multilosses.get_dice_from_alphas(float(alpha_1), float(alpha_2), float(alpha_3), float(alpha_4))

    alpha_1 = float(alpha_1)
    alpha_2 = float(alpha_2)
    alpha_3 = float(alpha_3) 
    alpha_4 = float(alpha_4)
    #global dice_val = 0
    print(alpha_1, " ", alpha_2," ",alpha_3," ",alpha_4)
    print("Total => ",alpha_1+alpha_2+alpha_3+alpha_4)
    
    model = DRRMSAN_multiscale_attention_bayes_022_attn_3(height=256, width=256, n_channels=3, alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_3 = alpha_3, alpha_4 = alpha_4)
    #model.summary()
    print(alpha_1, " ", alpha_2," ",alpha_3," ",alpha_4)

    # Here we will send the alphas to the actual model and in return
    # we will recieve the dice coefficient to optimise, since this is
    # a maximization problem, we return the -ve of objective function
    # to be maximized
    # dice_coef = drrmsan_multilosses.get_dice_from_alphas(float(alpha_1[0]), float(alpha_2[0]), float(alpha_3[0]), float(alpha_4[0]))
    # dice_coef =  float(alpha_1)+ float(alpha_2)+ float(alpha_3)+ float(alpha_4)

    opt = Adam(learning_rate=1e-5)
    metrics = [dice_coef, jaccard, Recall(), Precision() ,'accuracy']
    model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)



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


    print(len(x_train))
    print(len(y_train))

    


    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks,
        verbose=2
    )

    import pandas as pd
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save to json:
    hist_json_file = 'history_isic_drrmsan.json'
    # with open(hist_json_file, 'a') as out:
    #     out.write(hist_df.to_json())
    #     out.write(",")
    #     out.close()

    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    
    # or save to csv:
    hist_csv_file = 'history_isic_drrmsan.csv'
    # with open(hist_csv_file, 'a') as out:
    #     out.write(str(hist_df.to_csv()))
    #     out.write(",")
    #     out.close()


    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    
    model.save_weights("isic_drrmsan_150e.h5")
    model.save("isic_drrmsan_with_weight_150e.h5")

    # Run this module only while loading the pre-trained model.
    model = load_model('isic_drrmsan_with_weight_150e.h5',custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef, 'jaccard':jaccard})
    #model.summary()

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

    fp = open('models/log_drrmsan_isic.txt','w')
    fp.close()
    fp = open('models/best_drrmsan_isic.txt','w')
    fp.write('-1.0')
    fp.close()

    dice, jacard = evaluateModel(model, X_test, Y_test, BATCH)
    # del model, history, X_test, train_data, valid_data, Y_test

    # print("Model deleted!!")
    

    print("Dice = ",dice)
    print("Jacard = ",jacard)
    return -float(dice*jacard)




########################################################


(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(PATH)

print("Training data: ", len(x_train))
print("Validation data: ", len(x_val))
print("Testing data: ", len(x_test))





train_data = tf_dataset(x_train, y_train, batch=BATCH)
valid_data = tf_dataset(x_val, y_val, batch=BATCH)




domain = [{'name': 'alpha_1', 'type': 'continuous', 'domain': (0,1), 'dimensionality':1},
          {'name': 'alpha_2', 'type': 'continuous', 'domain': (0,1), 'dimensionality':1},
          {'name': 'alpha_3', 'type': 'continuous', 'domain': (0,1), 'dimensionality':1},
          {'name': 'alpha_4', 'type': 'continuous', 'domain': (0,1), 'dimensionality':1}]

constraints = [{'name': 'constr_1', 'constraint':  '0.998 - x[:,0] - x[:,1] - x[:,2] - x[:,3]'},
               {'name': 'constr_1', 'constraint': '-0.999 + x[:,0] + x[:,1] + x[:,2] + x[:,3]'}]



def load_entire_file_into_memory_and_then_convert(filename):
    with open(filename, 'r') as input_file:
        full_file_contents = input_file.read()
        lines_of_file = full_file_contents.split('\n')
        return np.array(lines_of_file)

dump = load_entire_file_into_memory_and_then_convert('store_alphas_dice.txt')
print(dump)

X = []
Y = []
for item in dump[:-1]:
    all_items = item.split(' ')
    X.append([float(i) for i in all_items[:4]]) 
    Y.append(float(all_items[-2]))

X = np.array(X)
print(X)

Y = -np.array(Y)
print(Y)

Y = np.expand_dims(Y, axis=1)
Y

maxiter = 1

kernel = GPy.kern.Matern52(input_dim=4, ARD=True, variance=1, lengthscale=[1,1,1,1]);

myBopt_4d = GPyOpt.methods.BayesianOptimization(f=f, X=X, Y=Y, domain=domain, constraints = constraints,
                                                kernel=kernel, acquisition_type ='EI', model_type='GP', 
                                                verbosity=True, acquisition_optimizer_type='lbfgs', 
                                                cost_withGradients=None, exact_feval=True)

myBopt_4d.run_optimization(max_iter = maxiter, verbosity=True)
print("="*20)
print("Value of (x,y) that minimises the objective:"+str(myBopt_4d.x_opt))    
print("Minimum value of the objective: "+str(myBopt_4d.fx_opt))     
print("="*20)
#myBopt_4d.plot_acquisition()

f = open("./bayesian_opt.txt", "a+")
dump_str = "Value of (x,y) that minimises the objective:"+str(myBopt_4d.x_opt)+"\n"
f.write(dump_str)
dump_str = "Minimum value of the objective: "+str(myBopt_4d.fx_opt)+"\n"
f.write(dump_str)
f.close()