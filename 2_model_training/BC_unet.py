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
    input_tensor = tf.keras.Input(shape=(image_size[0],image_size[1],image_size[2],image_size[3]),
                                    dtype=tf.dtypes.float32,
                                        name='input_layer')

    return input_tensor

def build_unet_input_block(input_tensor,args):
    
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
    
    count=0
    tensor = Conv3D(filters=16,
                    kernel_size=(conv_size, conv_size, conv_size),
                    kernel_regularizer=tf.keras.regularizers.l2(L2),
                    strides=stride,
                    activation=activation,
                    input_shape=input_tensor.shape,
                    padding=padding,
                    dtype=tf.dtypes.float32,
                    name='input_Conv_3D_%s_%s_%s_%s'%(activation,padding,conv_size,count))(input_tensor)
    if drop>0:
        tensor=SpatialDropout3D(rate=drop,
                                dtype=tf.dtypes.float32,
                                data_format='channels_last',
                                name='input_drop_%s_%s'%(drop,count))(tensor)
    count=1
    for i in range(num_conv-1):
        tensor = Conv3D(filters=16,
                        kernel_size=(conv_size,conv_size,conv_size),
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        strides=stride,
                        activation=activation,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='input_Conv_3D_%s_%s_%s_%s'%(activation,padding,conv_size,count))(tensor)
        if drop>0:
            tensor=SpatialDropout3D(rate=drop,
                            dtype=tf.dtypes.float32,
                            data_format='channels_last',
                            name='input_drop_%s_%s'%(drop,count))(tensor)
        count=count+1
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

    count=0
    tensor = Conv3D(filters=16,
                    kernel_size=(conv_size, conv_size, conv_size),
                    kernel_regularizer=tf.keras.regularizers.l2(L2),
                    strides=stride,
                    activation=activation,
                    input_shape=tensor.shape,
                    padding=padding,
                    dtype=tf.dtypes.float32,
                    name='output_Conv_3D_%s_%s_%s_%s'%(activation,padding,conv_size,count))(tensor)
    if drop>0:
        tensor=SpatialDropout3D(rate=drop,
                                data_format='channels_last',
                                dtype=tf.dtypes.float32,
                                name='output_drop_%s_%s'%(drop,count))(tensor)
    
    count=1
    for i in range(num_conv-1):
        tensor = Conv3D(filters=16,
                        kernel_size=(conv_size,conv_size,conv_size),
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        strides=stride,
                        activation=activation,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='outut_Conv_3D_%s_%s_%s_%s'%(activation,padding,conv_size,count))(tensor)
        if drop>0:
            tensor = SpatialDropout3D(rate=drop,
                                        dtype=tf.dtypes.float32,
                                        data_format='channels_last',
                                        name='output_drop_%s_%s'%(drop,count))(tensor)
        count=count+1
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

    print('encoder input shape:',tensor.shape)
    print('encoder filters:',filter_list)
    print('num Conv3D per layer:',num_conv)
    print('deep:',deep)

    conv3d_count = 0
    tensor_stack = []
    for f,filter in enumerate(filter_list):
        for n in range(num_conv):
            print(f,filter,n)
            tensor = Conv3D(filters=filter,
                        kernel_size=(conv_size,conv_size,conv_size),
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        strides=stride,
                        activation=activation,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='Enconder_Conv_3D_%s_%s_%s'%(activation,padding,conv3d_count))(tensor)
            if drop>0:
                tensor=SpatialDropout3D(rate=drop,
                                dtype=tf.dtypes.float32,
                                data_format='channels_last',
                                name='Encoder_Drop_3D_%s_%s'%(drop,conv3d_count))(tensor)
            if n==(num_conv-1):
                tensor_stack.append(tensor)
            conv3d_count+=1

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

    print("building decoder")
    print("are we skipping? ",skip)
    conv3d_count=0
    upsample_count=0
    encoder_tensor_stack.pop()
    for f,filter in enumerate(filter_list):
        for n in range(num_conv):
            print(f,filter,n)
            tensor = Conv3D(filters=filter,
                        kernel_size=(conv_size,conv_size,conv_size),
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        strides=stride,
                        activation=activation,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='Decoder_Conv_3D_%s_%s_%s'%(activation,padding,conv3d_count))(tensor)
            if drop>0:
                tensor=SpatialDropout3D(rate=drop,
                                        dtype=tf.dtypes.float32,
                                        data_format='channels_last',
                                        name='Decoder_Drop_3D_%s_%s'%(drop,conv3d_count))(tensor)
            
            
            conv3d_count=conv3d_count+1

        if f<=1:
            tensor = UpSampling3D(size=(pool_size,pool_size,pool_size),
                                    name='Decoder_UpSample_%s'%(upsample_count))(tensor)
            upsample_count=upsample_count+1

            if skip==True:
                tensor = Concatenate()([tensor, encoder_tensor_stack.pop()])
                tensor = Conv3D(filters=filter,
                        kernel_size=(conv_size,conv_size,conv_size),
                        kernel_regularizer=tf.keras.regularizers.l2(L2),
                        strides=stride,
                        activation=activation,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='Decoder_Conv_3D_%s_%s_%s'%(activation,padding,conv3d_count))(tensor)
                if drop>0:
                    tensor=SpatialDropout3D(rate=drop,
                                            data_format='channels_last',
                                            dtype=tf.dtypes.float32,
                                            name='Decoder_drop_%s_%s'%(drop,conv3d_count))(tensor)
                conv3d_count=conv3d_count+1

    return tensor

def create_stacked_unet(args):

    image_size = args.image_size
    activation_conv = args.activation_conv
    activation_last = args.activation_last
    conv_size = args.conv_size
    stride = args.stride
    padding=args.padding
    lrate = args.lrate
    loss = args.loss
    metrics = args.metrics
    

    #create the input layer and layer normalize
    input_tensor = create_unet_input_tensor(args)

    #create the input block to learning across the time dimension
    tensor = build_unet_input_block(input_tensor,args)

    # build the encoder
    tensor, encoder_tensor_stack = build_unet_encoder(args,tensor)#self declared function

    #build the decoder
    tensor = build_unet_decoder(args,tensor,encoder_tensor_stack)#self declared function

    #generate an additional layer to convolve the ouputs for symmetry
    tensor = build_unet_output_block(args,tensor)

    #generate the output layer (4-days of lightning)
    output_tensor = Conv3D(filters = 1,
                            input_shape=tensor.shape,
                            dtype=tf.dtypes.float32,
                            activation=activation_last,
                            strides=stride,
                            padding=padding,
                            kernel_size=(conv_size,conv_size,conv_size))(tensor)
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
    #plot_model(model, to_file='test_unet.png', show_shapes=True, show_layer_names=True)
    print(model.summary())