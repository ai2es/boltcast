import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.layers import InputLayer, Dense, Activation, Dropout, BatchNormalization, Concatenate, LayerNormalization
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, SpatialDropout2D, SpatialDropout3D, AveragePooling2D, UpSampling2D,UpSampling3D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from BC_parser import *

def create_unet_input_tensor(args): 

    print("building the input tensors")
    image_size = args.image_size
    print('image_size:',image_size)
    input_tensor = tf.keras.Input(shape=(image_size[0],image_size[1],image_size[2],image_size[3]),
                                    dtype=tf.dtypes.float32,
                                        name='input_layer')
    return input_tensor

def build_unet_input_block(tensor,args):
    
    #get the model parameters
    image_size = args.image_size
    padding=args.padding
    activation=args.activation_conv
    conv_size = args.conv_size
    pool_size = args.pool
    stride = args.stride
    L2 = args.L2_reg
    num_conv = args.n_conv_per_step
    drop = args.spatial_dropout
    
    for i in range(num_conv):
        kernel_time = tensor.shape[1]
        tensor = Conv3D(filters=16,
                        kernel_size=(kernel_time, conv_size, conv_size),
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        strides=stride,
                        activation=activation,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='input_block_%s_kernel_2d_%s_kernel_time_%s_num_layer_%s'%(activation,conv_size,kernel_time,i))(tensor)
    
    if drop>0:
        tensor=SpatialDropout3D(rate=drop,
                        dtype=tf.dtypes.float32,
                        data_format='channels_last',
                        name='input_block_drop_%s'%(drop))(tensor)
    return tensor

def build_unet_encoder(args,tensor):

    print("building the encoder")
    padding=args.padding
    activation=args.activation_conv
    conv_size = args.conv_size
    pool_size = args.pool
    stride = args.stride
    filter_list = args.conv_nfilters
    num_conv = args.n_conv_per_step
    bn = args.batch_normalization
    drop = args.spatial_dropout
    L2 = args.L2_reg
    deep = args.deep

    conv3d_count = 0
    tensor_stack = []
    for f,filter in enumerate(filter_list):
        for n in range(num_conv):
            print(f,filter,n)
            kernel_time = tensor.shape[1]
            tensor = Conv3D(filters=filter,
                        kernel_size=(kernel_time,conv_size,conv_size),
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        strides=stride,
                        activation=activation,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='Enconder_%s_2d_kernel_%s_kernel_time_%s_num_layer_%s'%(activation,conv_size,kernel_time,conv3d_count))(tensor)
            if n==(num_conv-1):
                tensor_stack.append(tensor)
            conv3d_count+=1
        
        if drop>0:
            tensor=SpatialDropout3D(rate=drop,
                                dtype=tf.dtypes.float32,
                                data_format='channels_last',
                                name='Encoder_Drop_3D_%s_%s'%(drop,f))(tensor)

        if f<=1:
            print("trying 3D pooling")
            print(tensor.shape)
            tensor = MaxPooling3D(pool_size = (pool_size,pool_size,pool_size))(tensor)

    return tensor, tensor_stack

def build_unet_decoder(args,tensor,encoder_tensor_stack):

    image_size = args.image_size
    activation = args.activation_conv
    conv_size = args.conv_size
    stride = args.stride
    padding=args.padding
    lrate = args.lrate
    skip = args.skip
    pool_size=args.pool
    filter_list = np.flip(args.conv_nfilters)
    num_conv = args.n_conv_per_step
    bn = args.batch_normalization
    drop = args.spatial_dropout
    L2 = args.L2_reg

    conv3d_count=0
    upsample_count=0
    encoder_tensor_stack.pop()
    for f,filter in enumerate(filter_list):
        for n in range(num_conv):
            print(f,filter,n)
            kernel_time = tensor.shape[1]
            tensor = Conv3D(filters=filter,
                        kernel_size=(kernel_time,conv_size,conv_size),
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        strides=stride,
                        activation=activation,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='Decoder_%s_kernel_2d_%s_kernel_time_%s_num_layer_%s'%(activation,conv_size,kernel_time,conv3d_count))(tensor)
            conv3d_count=conv3d_count+1
        if drop>0:
            tensor=SpatialDropout3D(rate=drop,
                                    dtype=tf.dtypes.float32,
                                    data_format='channels_last',
                                    name='Decoder_Drop_3D_%s_%s'%(drop,f))(tensor)
        if f<=1:
            tensor = UpSampling3D(size=(pool_size,pool_size,pool_size),
                                    name='Decoder_UpSample_%s'%(upsample_count))(tensor)
            upsample_count=upsample_count+1
            if skip==True:
                tensor = Concatenate()([tensor, encoder_tensor_stack.pop()])
    return tensor

def build_unet_output_block(args,tensor):

    #get the model parameters
    image_size = args.image_size
    padding=args.padding
    activation=args.activation_conv
    conv_size = args.conv_size
    pool_size = args.pool
    stride = args.stride
    L2 = args.L2_reg
    num_conv = args.n_conv_per_step
    drop=args.spatial_dropout

    for n in range(num_conv):
        kernel_time = tensor.shape[1]
        tensor = Conv3D(filters=16,
                        kernel_size=(kernel_time, conv_size, conv_size),
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        strides=stride,
                        activation=activation,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='output_block_%s_kernel_2d_%s_kernel_time_%s_%s'%(activation,conv_size,kernel_time,n))(tensor)
    if drop>0:
        tensor=SpatialDropout3D(rate=drop,
                                dtype=tf.dtypes.float32,
                                data_format='channels_last',
                                name='OutputBlock_Drop_3D_%s'%(drop))(tensor)
    return tensor

def create_stacked_unet(args):

    image_size = args.image_size
    activation_conv = args.activation_conv
    activation_last = args.activation_last
    conv_size = args.conv_size
    stride = args.stride
    padding=args.padding
    

    #create the input layer and layer normalize
    input_tensor = create_unet_input_tensor(args=args)
    print('input_tensor.shape[1]',input_tensor.shape[1])

    #create the input block to learning across the time dimension
    tensor = build_unet_input_block(tensor=input_tensor,args=args)

    # build the encoder
    tensor, encoder_tensor_stack = build_unet_encoder(args,tensor)#self declared function

    # #build the decoder
    tensor = build_unet_decoder(args,tensor,encoder_tensor_stack)#self declared function

    #generate an additional layer to convolve the ouputs for symmetry
    tensor = build_unet_output_block(args,tensor)

    #generate the output layer (4-days of lightning)
    kernel_time = tensor.shape[1]
    output_tensor = Conv3D(filters = 1,
                            input_shape=tensor.shape,
                            dtype=tf.dtypes.float32,
                            activation=activation_last,
                            strides=stride,
                            padding=padding,
                            kernel_size=(kernel_time,conv_size,conv_size))(tensor)
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

    model = create_stacked_unet(args)
    plot_model(model, to_file='unet_v2.png', show_shapes=True, show_layer_names=True)
    print(model.summary())