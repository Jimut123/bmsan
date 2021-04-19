#def get_dice_from_alphas(x):

dice = 0
jaccard = 0

def get_dice_from_alphas(alpha_1, alpha_2, alpha_3, alpha_4):
    """
    from numba import cuda
    cuda.select_device(0)
    cuda.close()
    print("="*40, "cuda closed, flushed")
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    alpha_1 = x[:, 0]
    alpha_2 = x[:, 1]
    alpha_3 = x[:, 2]
    alpha_4 = x[:, 3]
    """
    alpha_1 = float(alpha_1)
    alpha_2 = float(alpha_2)
    alpha_3 = float(alpha_3) 
    alpha_4 = float(alpha_4)
    #global dice_val = 0
    print(alpha_1, " ", alpha_2," ",alpha_3," ",alpha_4)
    print("Total => ",alpha_1+alpha_2+alpha_3+alpha_4)
    
    import os
    import numpy as np
    import cv2
    from glob import glob
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, recall_score
    from keras.optimizers import Adam
    from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
    from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
    from tensorflow.keras.models import Model , load_model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.metrics import Recall, Precision 
    from tensorflow.keras import backend as K
    import sys
    sys.path.insert(0, '../../')
    from models import DRRMSAN_multiscale_attention_bayes_022_attn_2
    
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

    def load_data(path, split=0.2):
        """
        from glob import glob
        images_list = sorted(glob(os.path.join(path, "trainx/*.bmp")))
        masks_list = sorted(glob(os.path.join(path, "trainy/*.bmp")))
        """

        import sys
        import glob
        from tqdm import tqdm
        
        ############################## 
        images_list = []
        masks_list = []

        img_files = glob.glob('../original_img/*.tif')
        msk_files = glob.glob('../ground_truth/*.tif')

        img_files.sort()
        msk_files.sort()

        print(len(img_files))
        print(len(msk_files))

        X = []
        Y = []

        for img_fl in tqdm(img_files):
            name = str(img_fl.split('.')[2]).split('/')[2]
            original_name = "../original_img/"+name+".tif"
            mask_name = "../ground_truth/"+name+"_mask.tif"
            if(img_fl.split('.')[-1]=='tif'):
                images_list.append(original_name)
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

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(PATH)

    print("Training data: ", len(x_train))
    print("Validation data: ", len(x_val))
    print("Testing data: ", len(x_test))


    def read_and_rgb(a):
        a = cv2.imread(a)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        return a
    
    
    model = DRRMSAN_multiscale_attention_bayes_022_attn_2(height=256, width=256, n_channels=3, alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_3 = alpha_3, alpha_4 = alpha_4)
    #model.summary()
    print(alpha_1, " ", alpha_2," ",alpha_3," ",alpha_4)


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
    hist_json_file = 'history_brain_mri_drrmsan.json'
    # with open(hist_json_file, 'a') as out:
    #     out.write(hist_df.to_json())
    #     out.write(",")
    #     out.close()

    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv:
    hist_csv_file = 'history_brain_mri_drrmsan.csv'
    # with open(hist_csv_file, 'a') as out:
    #     out.write(str(hist_df.to_csv()))
    #     out.write(",")
    #     out.close()


    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


    model.save_weights("brain_mri_drrmsan_150e.h5")
    model.save("brain_mri_drrmsan_with_weight_150e.h5")

    # Run this module only while loading the pre-trained model.
    model = load_model('brain_mri_drrmsan_with_weight_150e.h5',custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef, 'jacard':jacard})
    #model.summary()



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

        global dice, jaccard

        for i in range(10):
            try:
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

                jaccard = (np.sum(intersection)/np.sum(union))
                plt.suptitle('jaccard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jaccard))

                plt.savefig('results/'+str(i)+'.png',format='png')
                plt.close()
            except:
                pass
        
        jaccard = 0
        dice = 0
        avg_precision = 0
        recall_score = 0
        count = 0

        for i in range(len(Y_test)):

            yp_2 = yp[i].ravel()
            if np.sum(yp_2) > 0:
                count += 1
                y2 = Y_test[i].ravel()

                intersection = yp_2 * y2
                union = yp_2 + y2 - intersection
                avg_precision += average_precision_score(yp_2, y2)
                # recall_score += recall_score(yp_2, y2)

                jaccard += (np.sum(intersection)/np.sum(union))

                dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))


        jaccard /= count
        dice /= count
        avg_precision /= count
        # recall_score /= len(Y_test)


        print('jaccard Index : '+str(jaccard))
        print('Dice Coefficient : '+str(dice))
        with open("Output.txt", "a") as text_file:
            text_file.write("jaccard : {} Dice Coef : {} Avg. Precision : {}  \n".format( 
            str(jaccard), str(dice), str(avg_precision)))
        
        jaccard_index_list.append(jaccard)
        dice_coeff_list.append(dice)
        fp = open('models/log_drrmsan_brain_mri.txt','a')
        fp.write(str(jaccard)+'\n')
        fp.close()

        fp = open('models/best_drrmsan_brain_mri.txt','r')
        best = fp.read()
        fp.close()

        if(jaccard>float(best)):
            print('***********************************************')
            print('jaccard Index improved from '+str(best)+' to '+str(jaccard))
            print('***********************************************')
            fp = open('models/best_UNet_brain_mri.txt','w')
            fp.write(str(jaccard))
            fp.close()

            #saveModel(model)

        print("00"*50)
        f = open("./bayesian_opt_msan_2.txt", "a+")
        dump_str = str(alpha_1) + " " + str(alpha_2) + " " + str(alpha_3) + " " + str(alpha_4) + " " + str(dice) + " " + str(jaccard) + " " + str(dice*jaccard) + " \n"
        f.write(dump_str)
        f.close()
        print("Dice Value Used = ", -float(dice))
        del model
        print("Model deleted and dice value returned!!")


    from tqdm import tqdm

    X_test = []
    Y_test = []

    for img_fl, img_msk in tqdm(zip(x_test, y_test)):
        img = cv2.imread('{}'.format(img_fl), cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img,(256, 256), interpolation = cv2.INTER_CUBIC)
        X_test.append(resized_img)

        #img_msk = "../trainy/Y_img_"+str(img_fl.split('.')[2]).split('_')[-1]+".bmp"
        msk = cv2.imread('{}'.format(img_msk), cv2.IMREAD_GRAYSCALE)
        resized_msk = cv2.resize(msk,(256, 256), interpolation = cv2.INTER_CUBIC)
        Y_test.append(resized_msk)



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

    fp = open('models/log_drrmsan_brain_mri.txt','w')
    fp.close()
    fp = open('models/best_drrmsan_brain_mri.txt','w')
    fp.write('-1.0')
    fp.close()

    evaluateModel(model, X_test, Y_test, BATCH)
    del model, history, X_test, train_data, valid_data, Y_test

    print("Model deleted!!")

    #import json
    #import matplotlib.pyplot as plt

    #with open('history_skin_drrmsan.json', 'r') as f:
    #    array = json.load(f)
    #print (array)
    #print(json.dumps(array, indent=4, sort_keys=True))

    #for item in array:
    #    if 'val' in item and 'dice' in item and 'add' in item: # and 'activation' in item:#
    #        val = array[item]['9']
    #        print("Dice Value got = ",val)
    # return the -ve of dice value
    print(type(dice), type(jaccard))
    res = float(dice*jaccard)
    print("Dice = {} Jaccard = {} Res = {}".format(dice,jaccard,-res))
    return -float(res)




