"""
Super More concentration



"""
#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 28/02/2020 15:56
# @Author  : Jimut Bahan Pal
# DRRMSAN version with multi scaled alpha values
# Just minor changes in the attention module

from tqdm import tqdm
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add, ZeroPadding2D
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.metrics import Recall, Precision 


# https://stackoverflow.com/questions/55809286/how-to-create-a-custom-keras-layer-min-pooling-but-ignore-zeros
# Minpool2D implementation

def MinPooling2D(x, pool_size, strides):

    max_val = K.max(x) + 1 # we gonna replace all zeros with that value
    # replace all 0s with very high numbers
    is_zero = max_val * K.cast(K.equal(x,0), dtype=K.floatx())
    x = is_zero + x

    # execute pooling with 0s being replaced by a high number
    min_x = -K.pool2d(-x, pool_size=(2, 2), strides=(2, 2))

    # depending on the value we either substract the zero replacement or not
    is_result_zero = max_val * K.cast(K.equal(min_x, max_val), dtype=K.floatx()) 
    min_x = min_x - is_result_zero

    return min_x # concatenate on channel





##=================================================
from tensorflow.keras import layers
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    add, multiply, AveragePooling2D, SpatialDropout2D, Subtract, average
from tensorflow.keras import initializers
##=================================================




def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    # x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    # x = BatchNormalization(axis=3, scale=False)(x)

    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    # out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    # out = BatchNormalization(axis=3)(out)

    return out


def attention_up_and_concate(down_layer, layer, filters):
    '''
    Attention up and concatenate layer

    Arguments:
        down_layer {keras layer} -- layer coming from the down
        layer {keras layer} -- layer coming from the top
        filters {int} -- number of channels in image

    Returns:
        [keras layer] -- [output layer]
    '''
    
    # up = Conv2DTranspose(out_channel, [3,  3], strides=[3,  3])(down_layer)
    #up = UpSampling2D(size=(2, 2))(down_layer)

    layer = proposed_attention_block_2d(down_layer, layer, filters)

    # if data_format == 'channels_first':
    #     my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    # else:
    #     my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    #concate = my_concat([down_layer, layer])
    return layer



"""
# Attention - 3
def proposed_attention_block_2d(ms_conv, res_block, filters):
    '''
    Proposed Attention block - Version 3 <|smash-4|> 

    Arguments:
        ms_conv {keras layer} -- layer coming from the multi resolution convolution
        res_block {keras layer} -- layer coming from the residual block
        filters {int} -- number of channels in image

    Returns:
        [keras layer] -- [output layer]
    '''

    up_1 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(ms_conv))
    up_2 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(up_1))
    up_3 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(up_2))
    up_4 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(up_3))

    up_11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(up_1)
    up_22 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(up_2)
    up_33 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(up_3)
    up_44 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(up_4)

    down_1 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(res_block))
    down_2 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(down_1))
    down_3 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(down_2))
    down_4 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(down_3))

    down_11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(down_1)
    down_22 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(down_2)
    down_33 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(down_3)
    down_44 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(down_4)

    mult_block_1 = multiply([up_11, down_11])
    mult_block_2 = multiply([up_22, down_22])
    mult_block_3 = multiply([up_33, down_33])
    mult_block_4 = multiply([up_44, down_44])

    attn_map = Activation('sigmoid')(add([mult_block_1, mult_block_2, mult_block_3, mult_block_4]))

    attn_upsampled =  UpSampling2D(size=(2, 2))(attn_map)

    attn_output_1 = multiply([attn_upsampled, res_block])

    return attn_output_1
"""


# Attention - 2
def proposed_attention_block_2d(ms_conv, res_block, filters):
    '''
    Proposed Attention block - Version 2.1 <|dream-004|>

    Arguments:
        ms_conv {keras layer} -- layer coming from the multi resolution convolution
        res_block {keras layer} -- layer coming from the residual block
        filters {int} -- number of channels in image

    Returns:
        [keras layer] -- [output layer]
    '''

    up_1 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(ms_conv))
    up_2 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(up_1))
    up_3 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(up_2))
    up_4 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(up_3))
    

    down_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(up_1)
    down_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(up_2)
    down_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(up_3)
    down_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(up_4)
    
    mult_block = Activation('sigmoid')(add([down_1, down_2, down_3, down_4]))

    up_attn = ZeroPadding2D(padding=(1,1))(UpSampling2D(size=(2, 2))(mult_block))

    attn_output_1 = multiply([up_attn, res_block])
    
    return attn_output_1


"""
def proposed_attention_block_2d(ms_conv, res_block, filters):
    '''
    Proposed Attention block

    Arguments:
        ms_conv {keras layer} -- layer coming from the multi resolution convolution
        res_block {keras layer} -- layer coming from the residual block
        filters {int} -- number of channels in image

    Returns:
        [keras layer] -- [output layer]
    '''

    theta_x = Conv2D(filters, [2,  2], strides=[1, 1], padding='same')(ms_conv)
    joint_conv_2x2 = Conv2D(filters, (2, 2), strides=(1, 1), padding='same')(theta_x)
    conv_3x3 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(joint_conv_2x2))
    conv_5x5 = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(joint_conv_2x2)
    conv_5x5 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(conv_5x5))
    conv_7x7 = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(joint_conv_2x2)
    conv_7x7 = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(conv_7x7)
    conv_7x7 = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(conv_7x7))
    add_3x3_5x5 = add([conv_3x3, conv_5x5])
    mult_3x3_5x5 = multiply([conv_3x3, conv_5x5]) #multiply([conv_3x3, conv_5x5])#Subtract()([conv_3x3, conv_5x5])
    add_3x3_5x5_7x7 = Activation('relu')(add([add_3x3_5x5, conv_7x7]))
    mul_3x3_5x5_7x7 = Activation('relu')(multiply([mult_3x3_5x5, conv_7x7]))#multiply([mult_3x3_5x5, conv_7x7]))) # Subtract()([mult_3x3_5x5, conv_7x7])
    add_1x1_upper = Activation('relu')(Conv2D(filters, [2,  2], strides=[1, 1], padding='same')(add_3x3_5x5_7x7))
    mult_1x1_lower = Activation('relu')(Conv2D(filters, [2,  2], strides=[1, 1], padding='same')(mul_3x3_5x5_7x7))
    resampler_down_upper = MaxPooling2D(pool_size=(12, 12), strides=(2, 2))(add_1x1_upper) #AveragePooling2D
    resampler_down_lower = MaxPooling2D(pool_size=(12, 12), strides=(2, 2))(mult_1x1_lower)
    output_ms_conv_res_block = multiply([resampler_down_upper, resampler_down_lower])

    theta_x_rb = Conv2D(filters, [2,  2], strides=[1, 1], padding='same')(res_block)
    joint_conv_2x2_rb = Conv2D(filters, (2, 2), strides=(1, 1), padding='same')(theta_x_rb)
    conv_3x3_rb = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(joint_conv_2x2_rb))
    conv_5x5_rb = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(joint_conv_2x2_rb)
    conv_5x5_rb = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(conv_5x5_rb))
    conv_7x7_rb = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(joint_conv_2x2_rb)
    conv_7x7_rb = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(conv_7x7_rb)
    conv_7x7_rb = Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(conv_7x7_rb))
    add_3x3_5x5_rb = add([conv_3x3_rb, conv_5x5_rb])
    mult_3x3_5x5_rb = multiply([conv_3x3_rb, conv_5x5_rb])#multiply([conv_3x3_rb, conv_5x5_rb]) #Subtract()([conv_3x3_rb, conv_5x5_rb])
    add_3x3_5x5_7x7_rb = Activation('relu')(add([add_3x3_5x5_rb, conv_7x7_rb]))
    mul_3x3_5x5_7x7_rb = Activation('relu')(multiply([mult_3x3_5x5_rb, conv_7x7_rb]))#multiply([mult_3x3_5x5_rb, conv_7x7_rb]))) # Subtract()([mult_3x3_5x5_rb, conv_7x7_rb])
    add_1x1_upper_rb = Activation('relu')(Conv2D(filters, [2,  2], strides=[1, 1], padding='same')(add_3x3_5x5_7x7_rb))
    mult_1x1_lower_rb = Activation('relu')(Conv2D(filters, [2, 2], strides=[1, 1], padding='same')(mul_3x3_5x5_7x7_rb))
    resampler_down_upper_rb = MaxPooling2D(pool_size=(12, 12), strides=(2, 2))(add_1x1_upper_rb)
    resampler_down_lower_rb = MaxPooling2D(pool_size=(12, 12), strides=(2, 2))(mult_1x1_lower_rb)
    output_ms_conv_res_block_rb = multiply([resampler_down_upper_rb, resampler_down_lower_rb])
    
    attn_outputs_mult = Activation('sigmoid')(multiply([output_ms_conv_res_block, output_ms_conv_res_block_rb]))
    attn_output_1 = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(ZeroPadding2D(padding=(5,5))(UpSampling2D(size=(2, 2))(attn_outputs_mult)))
    
    return attn_output_1
"""

def ResPath(filters, length, inp):
    '''
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    # out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        # out = BatchNormalization(axis=3)(out)

    return out


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1], padding='same'):
    skip_layer = input_layer
    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(filters, kernel_size, strides=stride, padding=padding)(layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(filters, kernel_size, strides=stride, padding=padding)(add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1
    out_layer = add([layer, skip_layer])
    return out_layer



def DRRMSAN_multiscale_wo_ms_wo_lo(height, width, n_channels, alpha_1, alpha_2, alpha_3, alpha_4):
    '''
    DRRMSAN Multiscale Attention Model

    Arguments:
        height {int} -- height of image
        width {int} -- width of image
        n_channels {int} -- number of channels in image

    Returns:
        [keras model] -- DRRMSAN model
    '''
    print("DRRMSAN Bayes")

    inputs = Input((height, width, n_channels))

    total_1_2I = 51
    total_1_4I = 105
    total_1_8I = 212

    #==================================================================

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    #===================
    pool1 = Conv2D(total_1_2I, (3, 3), strides=(1,1), padding='same')(pool1)
    #===================
    mresblock1 = ResPath(32, 4, mresblock1)

    
    mresblock2 = MultiResBlock(32*2, pool1)
    #mresblock2 = rec_res_block(mresblock2, 105)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    #===================
    pool2 = Conv2D(total_1_4I, (3, 3), strides=(1,1), padding='same')(pool2)
    #===================
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    #===================
    pool3 = Conv2D(total_1_8I, (3, 3), strides=(1,1), padding='same')(pool3)
    #===================
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)


    mresblock5 = MultiResBlock(32*16, pool4)
    up6 = proposed_attention_block_2d(Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same', name='up6')(mresblock5), mresblock4,filters=256)
    up6 = concatenate([up6, mresblock4])

    mresblock6 = MultiResBlock(32*8, up6)
    
    conv_6_up = Conv2D(212, (3, 3), padding='same', activation='relu', name='conv_6_up')(mresblock6)


    up7 = proposed_attention_block_2d(Conv2DTranspose(32*4, (2, 2), strides=(2, 2), padding='same', name='up7')(mresblock6), mresblock3, filters = 32*4)
    up7 = concatenate([up7, mresblock3])

    mresblock7 = MultiResBlock(32*4, up7)
    
    conv_7_up = Conv2D(105, (3, 3), padding='same', activation='relu', name='conv_7_up')(mresblock7)
    
    
    up8 = proposed_attention_block_2d(Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same', name='up8')(mresblock7), mresblock2, filters = 32*2)
    up8 = concatenate([up8, mresblock2])#,

    mresblock8 = MultiResBlock(32*2, up8)
    
    conv_8_up = Conv2D(51, (3, 3), padding='same', activation='relu', name='conv_8_up')(mresblock8)
    
    up9 = proposed_attention_block_2d(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='up9')(mresblock8), mresblock1, filters = 32)
    up9 = concatenate([up9, mresblock1])#

    mresblock9 = MultiResBlock(32, up9)
    
    conv_9_up = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_8_up')(mresblock9)
    


    side6 = UpSampling2D(size=(8, 8))(conv_6_up)
    side7 = UpSampling2D(size=(4, 4))(conv_7_up)
    side8 = UpSampling2D(size=(2, 2))(conv_8_up)

    # the conv blocks on the right sides

    out6 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='side_6')(side6) 
    out7 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='side_7')(side7) 
    out8 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='side_8')(side8) 
    out9 = conv2d_bn(mresblock9, 1, 3, 3, activation='sigmoid', padding='same')

    
    out10 = add([alpha_1 * out6, alpha_2 * out7, alpha_3 * out8, alpha_4 * out9])

    

    model = Model(inputs=[inputs], outputs=[out10])

    return model