import argparse
from keras.layers import Input, Conv2D, Conv2DTranspose, Conv3D, ConvLSTM2D, MaxPooling2D, MaxPooling3D, UpSampling3D, TimeDistributed, Concatenate, SpatialDropout3D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import keras
import numpy as np
from BC_parser import *
import keras_tuner as kt

def create_LSTM(args):
    lrate = args.lrate
    image_size = args.image_size
    padding = args.lstm_padding
    activation = args.lstm_conv_activation
    activation_last = args.lstm_activation_last
    conv_size = args.lstm_conv_size
    pool = args.lstm_pool
    ret_state = args.return_state
    ret_seq = args.return_sequences
    num_filters = 16
    L2 = args.L2_reg
    drop = args.spatial_dropout

    input_tensor = Input(shape=(image_size[0],image_size[1],image_size[2],image_size[3]),
                                    dtype=tf.dtypes.float32,
                                    name='input')
    tensor = input_tensor
    kernel_time = tensor.shape[1]
    kernel_size = (kernel_time,conv_size,conv_size)
    tensor = Conv3D(filters=num_filters,
                        input_shape = tensor.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        kernel_size = kernel_size,
                        name='encoder_kernel_2d_%s_kernel_time_%s_activation_%s'%(conv_size,kernel_time,activation))(tensor)
    tensor = SpatialDropout3D(rate=drop,
                                dtype=tf.dtypes.float32,
                                name='drop_encoder')(tensor)
    
    ##########################################CONV_LSTM_Layers#######################################################
    print("building the ConvLSTM layer")
    kernel_size = (conv_size,conv_size)
    tensor, h_tensor, c_tensor = ConvLSTM2D(filters=num_filters, 
                            kernel_size=kernel_size, 
                            padding=padding, 
                            return_sequences=ret_seq,
                            return_state=ret_state,
                            kernel_regularizer=tf.keras.regularizers.l2(L2),
                            input_shape = tensor.shape,
                            name='convlstm2d_kernel_2d_%s'%(conv_size))(tensor)
    ######################################################################################################################
    kernel_time = tensor.shape[1]
    kernel_size = (kernel_time,conv_size,conv_size)
    tensor = Conv3D(filters=num_filters,
                        input_shape=tensor.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        kernel_size=kernel_size,
                        name='decoder_kernel_2d_%s_kernel_time_%s_activation_%s'%(conv_size,kernel_time,activation))(tensor)
    tensor = SpatialDropout3D(rate=drop,
                                dtype=tf.dtypes.float32,
                                name='drop_decoder')(tensor)

    kernel_time = tensor.shape[1]
    kernel_size = (kernel_time,conv_size,conv_size)
    output_tensor = Conv3D(filters=1,
                        input_shape=tensor.shape,
                        activation=activation_last,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        kernel_size=kernel_size,
                        name='output_layer_kernel_2d_%s_kernel_time_%s_activation_%s'%(conv_size,kernel_time,activation))(tensor)
    
    #complete the model
    model = Model(inputs=input_tensor,outputs=output_tensor)
    return model

if __name__ == "__main__":

    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print(n_visible_devices)
    tf.config.set_visible_devices([], 'GPU')
    print('NO VISIBLE DEVICES!!!!')

    parser = create_parser()
    args = parser.parse_args()
    print(args)

    model = create_LSTM(args)
    # plot_model(model, to_file='lstm_v2.png', show_shapes=True, show_layer_names=True)
    print(model.summary())