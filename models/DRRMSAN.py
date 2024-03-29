#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 01/12/2020 15:56
# @Author  : Jimut Bahan Pal

from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
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
    x = BatchNormalization(axis=3, scale=False)(x)

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
    x = BatchNormalization(axis=3, scale=False)(x)

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
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out

def attention_up_and_concate(down_layer, layer, filters):
    
    # up = Conv2DTranspose(out_channel, [3,  3], strides=[3,  3])(down_layer)
    #up = UpSampling2D(size=(2, 2))(down_layer)

    layer = proposed_attention_block_2d(down_layer, layer, filters)

    # if data_format == 'channels_first':
    #     my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    # else:
    #     my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    #concate = my_concat([down_layer, layer])
    return layer

def attention_block_2d(ms_conv, res_block, filters):
    # theta_x(?,g_height,g_width,filters)
    theta_x = Conv2D(filters, [1, 1], strides=[1, 1])(ms_conv)

    # phi_g(?,g_height,g_width,filters)
    phi_g = Conv2D(filters, [1, 1], strides=[1, 1])(res_block)

    # f(?,g_height,g_width,filters)
    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)
    # att_x(?,x_height,x_width,x_channel)
    att_x = multiply([ms_conv, rate])

    return att_x


def proposed_attention_block_2d(ms_conv, res_block, filters):
    
    theta_x = Conv2D(filters, [1,  1], strides=[1, 1], padding='same')(ms_conv)
    joint_conv_2x2 = Conv2D(filters, (2, 2), strides=(1, 1), padding='same')(theta_x)
    conv_3x3 = SpatialDropout2D(0.5)(Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(joint_conv_2x2)))
    conv_5x5 = SpatialDropout2D(0.5)(Activation('relu')(Conv2D(filters, (5, 5), strides=(1, 1), padding='same')(joint_conv_2x2)))
    conv_7x7 = SpatialDropout2D(0.5)(Activation('relu')(Conv2D(filters, (7, 7), strides=(1, 1), padding='same')(joint_conv_2x2)))
    add_3x3_5x5 = add([conv_3x3, conv_5x5])
    mult_3x3_5x5 = multiply([conv_3x3, conv_5x5]) #multiply([conv_3x3, conv_5x5])#Subtract()([conv_3x3, conv_5x5])
    add_3x3_5x5_7x7 = Activation('sigmoid')(add([add_3x3_5x5, conv_7x7]))
    mul_3x3_5x5_7x7 = Activation('sigmoid')(multiply([mult_3x3_5x5, conv_7x7]))#multiply([mult_3x3_5x5, conv_7x7]))) # Subtract()([mult_3x3_5x5, conv_7x7])
    add_1x1_upper = Activation('sigmoid')(Conv2D(filters, [1,  1], strides=[1, 1], padding='same')(add_3x3_5x5_7x7))
    mult_1x1_lower = Activation('sigmoid')(Conv2D(filters, [1,  1], strides=[1, 1], padding='same')(mul_3x3_5x5_7x7))
    resampler_down_upper = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(add_1x1_upper) #AveragePooling2D
    resampler_down_lower = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mult_1x1_lower)
    output_ms_conv_res_block = multiply([resampler_down_upper, resampler_down_lower])

    theta_x_rb = Conv2D(filters, [1,  1], strides=[1, 1], padding='same')(res_block)
    joint_conv_2x2_rb = Conv2D(filters, (2, 2), strides=(1, 1), padding='same')(theta_x_rb)
    conv_3x3_rb = SpatialDropout2D(0.5)(Activation('relu')(Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(joint_conv_2x2_rb)))
    conv_5x5_rb = SpatialDropout2D(0.5)(Activation('relu')(Conv2D(filters, (5, 5), strides=(1, 1), padding='same')(joint_conv_2x2_rb)))
    conv_7x7_rb = SpatialDropout2D(0.5)(Activation('relu')(Conv2D(filters, (7, 7), strides=(1, 1), padding='same')(joint_conv_2x2_rb)))
    add_3x3_5x5_rb = add([conv_3x3_rb, conv_5x5_rb])
    mult_3x3_5x5_rb = multiply([conv_3x3_rb, conv_5x5_rb])#multiply([conv_3x3_rb, conv_5x5_rb]) #Subtract()([conv_3x3_rb, conv_5x5_rb])
    add_3x3_5x5_7x7_rb = Activation('sigmoid')(add([add_3x3_5x5_rb, conv_7x7_rb]))
    mul_3x3_5x5_7x7_rb = Activation('sigmoid')(multiply([mult_3x3_5x5_rb, conv_7x7_rb]))#multiply([mult_3x3_5x5_rb, conv_7x7_rb]))) # Subtract()([mult_3x3_5x5_rb, conv_7x7_rb])
    add_1x1_upper_rb = Activation('sigmoid')(Conv2D(filters, [1,  1], strides=[1, 1], padding='same')(add_3x3_5x5_7x7_rb))
    mult_1x1_lower_rb = Activation('sigmoid')(Conv2D(filters, [1, 1], strides=[1, 1], padding='same')(mul_3x3_5x5_7x7_rb))
    resampler_down_upper_rb = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(add_1x1_upper_rb)
    resampler_down_lower_rb = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mult_1x1_lower_rb)
    output_ms_conv_res_block_rb = multiply([resampler_down_upper_rb, resampler_down_lower_rb])
    
    attn_outputs_mult = Activation('sigmoid')(multiply([output_ms_conv_res_block, output_ms_conv_res_block_rb]))
    attn_output_1 = UpSampling2D(size=(2, 2))(attn_outputs_mult)
    attn_output = multiply([attn_output_1, theta_x_rb])
    return attn_output


    

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
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def DRRMSAN_multiscale_attention(height, width, n_channels):
    '''
    DRRMSAN Multiscale Attention Model

    Arguments:
        height {int} -- height of image
        width {int} -- width of image
        n_channels {int} -- number of channels in image

    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, n_channels))

    # use average pool, maxpool and minpool to create different volumes of
    # multiscaling, minpool is used here as a sort of regularizer noise in the feature
    # space.  1/2 th the original scale first.

    inp_1_2I = AveragePooling2D(pool_size=(2, 2))(inputs)
    inp_1_2I_mxpool = MaxPooling2D(pool_size=(2, 2))(inputs)
    inp_1_2I_minpool = MinPooling2D(inputs, pool_size=(2,2), strides=(1,1))
    """
    tf.image.resize(
              inputs, (int(height * 1/2), int(width * 1/2)), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
              antialias=False, name=None
              )
    """
    # 1/4 rth the original scale
    inp_1_4I = AveragePooling2D(pool_size=(2, 2))(inp_1_2I)
    inp_1_4I_mxpool = MaxPooling2D(pool_size=(2, 2))(inp_1_2I_mxpool)
    inp_1_4I_minpool = MinPooling2D(inp_1_2I_minpool, pool_size=(2,2), strides=(1,1))
    #inp_1_4I_minpool = MaxPooling2D(pool_size=(2, 2))(inp_1_2I_mxpool)
    """
    tf.image.resize(
              inputs, (int(height * 1/4), int(width * 1/4)), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
              antialias=False, name=None
              )
    """
    # 1/8 th the original scale
    inp_1_8I = AveragePooling2D(pool_size=(2, 2))(inp_1_4I)
    inp_1_8I_mxpool = MaxPooling2D(pool_size=(2, 2))(inp_1_4I_mxpool)
    inp_1_8I_minpool = MinPooling2D(inp_1_4I_minpool, pool_size=(2,2), strides=(1,1))
    """
              tf.image.resize(
              inputs, (int(height * 1/8), int(width * 1/8)), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
              antialias=False, name=None
              )
    """
    # just pass through some conv and add
    # for adding to multi res block 2, 32 filters
    # use 50 - 50 
    # Conv2D(filters, (3, 3), strides=(1,1), padding='same'
    
    # using different ratios for the volumes, can be improved by using
    # Bayesian Optimization

    total_1_2I = 51
    per_mx_pool_1_2I = int(0.40 * total_1_2I)
    per_avg_pool_1_2I = int(0.30 * total_1_2I)
    per_min_pool_1_2I = int(0.05 * total_1_2I)
    per_down_1_2I = int(total_1_2I - (per_mx_pool_1_2I + per_avg_pool_1_2I + per_min_pool_1_2I))

    mrb2_1_2I_avgpool = Conv2D(per_avg_pool_1_2I, (3, 3), strides=(1,1), padding='same', name='side_left_1_avgpool')(inp_1_2I) 
    mrb2_1_2I_mxpool = Conv2D(per_mx_pool_1_2I, (3, 3), strides=(1,1), padding='same', name='side_left_1_mxpool')(inp_1_2I_mxpool)
    mrb2_1_2I_minpool = Conv2D(per_min_pool_1_2I, (3, 3), strides=(1,1), padding='same', name='side_left_1_minpool')(inp_1_2I_minpool)

    total_1_4I = 105
    per_mx_pool_1_4I = int(0.40 * total_1_4I)
    per_avg_pool_1_4I = int(0.30 * total_1_4I)
    per_min_pool_1_4I = int(0.05 * total_1_4I)
    # 52% to the down layer
    per_down_1_4I = int(total_1_4I - (per_mx_pool_1_4I + per_avg_pool_1_4I + per_min_pool_1_4I))

    mrb3_1_4I_avgpool = Conv2D(per_avg_pool_1_4I, (3, 3), strides=(1,1), padding='same', name='side_left_2_avgpool')(inp_1_4I) 
    mrb3_1_4I_mxpool = Conv2D(per_mx_pool_1_4I, (3, 3), strides=(1,1), padding='same', name='side_left_2_mxpool')(inp_1_4I_mxpool) 
    mrb3_1_4I_minpool = Conv2D(per_min_pool_1_4I, (3, 3), strides=(1,1), padding='same', name='side_left_2_minpool')(inp_1_4I_minpool) 

    total_1_8I = 212
    per_mx_pool_1_8I = int(0.40 * total_1_8I)
    per_avg_pool_1_8I = int(0.30 * total_1_8I)
    per_min_pool_1_8I = int(0.05 * total_1_8I)
    per_down_1_8I = int(total_1_8I - (per_mx_pool_1_8I + per_avg_pool_1_8I + per_min_pool_1_8I))

    mrb4_1_8I_avgpool = Conv2D(per_avg_pool_1_8I, (3, 3), strides=(1,1), padding='same', name='side_left_3_avgpool')(inp_1_8I)
    mrb4_1_8I_mxpool = Conv2D(per_mx_pool_1_8I, (3, 3), strides=(1,1), padding='same', name='side_left_3_mxpool')(inp_1_8I_mxpool)
    mrb4_1_8I_minpool = Conv2D(per_min_pool_1_8I, (3, 3), strides=(1,1), padding='same', name='side_left_3_minpool')(inp_1_8I_minpool)


    #==================================================================

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    #===================
    pool1 = Conv2D(per_down_1_2I, (3, 3), strides=(1,1), padding='same')(pool1)
    pool1 = concatenate([pool1, mrb2_1_2I_avgpool, mrb2_1_2I_mxpool, mrb2_1_2I_minpool])
    #pool1 = multiply([pool1, mrb2_1_2I])
    #pool1 = proposed_attention_block_2d(pool1, mrb2_1_2I,filters=51)
    #===================
    mresblock1 = ResPath(32, 4, mresblock1)

    
    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    #===================
    pool2 = Conv2D(per_down_1_4I, (3, 3), strides=(1,1), padding='same')(pool2)
    pool2 = concatenate([pool2, mrb3_1_4I_avgpool, mrb3_1_4I_mxpool, mrb3_1_4I_minpool ])
    #pool2 = multiply([pool2, mrb3_1_4I])
    #pool2 = proposed_attention_block_2d(pool2, mrb3_1_4I,filters=105)
    #===================
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    #===================
    pool3 = Conv2D(per_down_1_8I, (3, 3), strides=(1,1), padding='same')(pool3)
    pool3 = concatenate([pool3, mrb4_1_8I_avgpool, mrb4_1_8I_mxpool, mrb4_1_8I_minpool])
    #pool3 = multiply([pool3, mrb4_1_8I])
    #pool3 = proposed_attention_block_2d(pool3, mrb4_1_8I,filters=212)
    #===================
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6_add =  add([Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4])
    up6_dra = attention_up_and_concate(Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same', name='up6_dra')(mresblock5), mresblock4,filters=32*8)
    up6 = attention_block_2d(Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same', name='up6')(mresblock5), mresblock4,filters=64)
    up6 = add([up6, up6_add, up6_dra])
    
    #concatenate([Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)
    conv_6_up = Conv2D(212, (3, 3), padding='same', activation='relu', name='conv_6_up')(mresblock6)

    up7_add = add([Conv2DTranspose(32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3])
    up7_dra = attention_up_and_concate(Conv2DTranspose(32*4, (2, 2), strides=(2, 2), padding='same', name='up7_dra')(mresblock6), mresblock3, filters = 32*4)
    up7 = attention_block_2d(Conv2DTranspose(32*4, (2, 2), strides=(2, 2), padding='same', name='up7')(mresblock6), mresblock3, filters = 32*4)
    up7 = add([up7, up7_add, up7_dra])#,
    mresblock7 = MultiResBlock(32*4, up7)
    conv_7_up = Conv2D(105, (3, 3), padding='same', activation='relu', name='conv_7_up')(mresblock7)

    up8_add = add([Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2])
    up8_dra = attention_up_and_concate(Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same', name='up8_dra')(mresblock7), mresblock2, filters = 32*2)
    up8 = attention_block_2d(Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same', name='up8')(mresblock7), mresblock2, filters = 32*2)
    up8 = add([up8, up8_add, up8_dra])#,
    mresblock8 = MultiResBlock(32*2, up8)
    conv_8_up = Conv2D(51, (3, 3), padding='same', activation='relu', name='conv_8_up')(mresblock8)

    up9_add = add([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1])
    up9_dra = attention_up_and_concate(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='up9_dra')(mresblock8), mresblock1, filters = 32)
    up9 = attention_block_2d(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='up9')(mresblock8), mresblock1, filters = 32)
    up9 = add([up9, up9_add, up9_dra])#
    mresblock9 = MultiResBlock(32, up9)
    conv_9_up = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_8_up')(mresblock9)


    side6 = UpSampling2D(size=(8, 8))(conv_6_up)
    side7 = UpSampling2D(size=(4, 4))(conv_7_up)
    side8 = UpSampling2D(size=(2, 2))(conv_8_up)

    # the conv blocks on the right sides

    out6 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='side_6')(side6) # conv2d_bn(side6, 1, 1, 1, activation='none') #
    out7 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='side_7')(side7) # conv2d_bn(side7, 1, 1, 1, activation='none') #
    out8 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='side_8')(side8) # conv2d_bn(side8, 1, 1, 1, activation='none') #

    out9 = conv2d_bn(mresblock9, 1, 3, 3, activation='sigmoid', padding='same')

    # averaging all the output masks obtained at different scales

    out10 = average([out6, out7, out8, out9])

    #conv10 = conv2d_bn(out10, 1, 1, 1, activation='sigmoid')

    model = Model(inputs=[inputs], outputs=[out10])

    return model
