import keras_tuner as kt
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.layers import InputLayer, Dense, Activation, Dropout, BatchNormalization, Concatenate, LayerNormalization
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, SpatialDropout2D, SpatialDropout3D, AveragePooling2D, UpSampling2D,UpSampling3D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from BC_parser import *

def build_model(hp):
    print('building the model')
    model = keras.Sequential()
    model.add(keras.layers.Dense(
                hp.Choice('units', [8, 16, 32]),
                activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse')
    print('model compiled successfully')
    return model

def build_unet(hp):
    print("building the input tensors")
    image_size = [4,128,256,9]
    padding='same'
    pool_size = 2
    stride = 1
    num_conv = 3
    lrate=.00001
    filter_list = [32,64,128]
    act_hyper = hp.Choice("activation",['elu','relu'])
    l2_hyper = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5, 0.0])
    dropout_hyper = hp.Float("dropout",min_value=0.0,max_value=0.1,step=0.05)
    kernel_hyper = hp.Choice("kernel_2d", [2, 3, 4])
    skip=True

    print('image_size:',image_size)
    input_tensor = tf.keras.Input(shape=(image_size[0],image_size[1],image_size[2],image_size[3]),
                                    dtype=tf.dtypes.float32,
                                        name='input_layer')
    tensor=input_tensor
    #start input block
    for i in range(num_conv):
        kernel_time = tensor.shape[1]
        kernel_size = (kernel_time, kernel_hyper, kernel_hyper)
        tensor = Conv3D(filters=16,
                        kernel_size=kernel_size,
                        kernel_regularizer=keras.regularizers.l2(l2_hyper),
                        strides=stride,
                        activation=act_hyper,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='input_block_%s'%(i))(tensor)
    tensor = SpatialDropout3D(rate=dropout_hyper)(tensor)
    #end input block

    #start encoder
    conv3d_count = 0
    tensor_stack = []
    for f,filter in enumerate(filter_list):
        for n in range(num_conv):
            kernel_time = tensor.shape[1]
            kernel_size = (kernel_time,kernel_hyper,kernel_hyper)
            tensor = Conv3D(filters=filter,
                        kernel_size=kernel_size,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_hyper),
                        strides=stride,
                        activation=act_hyper,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='Enconder_%s_%s'%(f,n))(tensor)
            if n==(num_conv-1):
                tensor_stack.append(tensor)
            conv3d_count+=1
        #end block
        tensor=SpatialDropout3D(rate=dropout_hyper,
                            dtype=tf.dtypes.float32,
                            data_format='channels_last',
                            name='Encoder_Drop_3D_%s'%(f))(tensor)
        if f<=1:
            print("trying 3D pooling")
            print(tensor.shape)
            tensor = MaxPooling3D(pool_size = (pool_size,pool_size,pool_size))(tensor)
    #end encoder

    #begin decoder
    conv3d_count=0
    upsample_count=0
    tensor_stack.pop()
    filter_list = np.flip(filter_list)
    for f,filter in enumerate(filter_list):
        for n in range(num_conv):
            kernel_time = tensor.shape[1]
            kernel_size = (kernel_time, kernel_hyper, kernel_hyper)
            tensor = Conv3D(filters=filter,
                        kernel_size=kernel_size,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_hyper),
                        strides=stride,
                        activation=act_hyper,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='Decoder_%s_%s'%(f,n))(tensor)
            conv3d_count=conv3d_count+1

        tensor=SpatialDropout3D(rate=dropout_hyper,
                                dtype=tf.dtypes.float32,
                                data_format='channels_last',
                                name='Decoder_Drop_3D_%s'%(f))(tensor)
        if f<=1:
            tensor = UpSampling3D(size=(pool_size,pool_size,pool_size),
                                    name='Decoder_UpSample_%s'%(upsample_count))(tensor)
            upsample_count=upsample_count+1
            if skip==True:
                tensor = Concatenate()([tensor, tensor_stack.pop()])
    #end decoder

    #begin output block
    for n in range(num_conv):
        kernel_time = tensor.shape[1]
        kernel_size = (kernel_time, kernel_hyper, kernel_hyper)
        tensor = Conv3D(filters=16,
                        kernel_size=kernel_size,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_hyper),
                        strides=stride,
                        activation=act_hyper,
                        input_shape=tensor.shape,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        name='output_block_%s'%(n))(tensor)
    tensor = SpatialDropout3D(rate=dropout_hyper,
                            dtype=tf.dtypes.float32,
                            data_format='channels_last',
                            name='OutputBlock_Drop_3D')(tensor)
    #end output block

    #end output tensor
    kernel_size = (tensor.shape[1],kernel_hyper,kernel_hyper)
    tensor = Conv3D(filters = 1,
                            input_shape=tensor.shape,
                            dtype=tf.dtypes.float32,
                            activation=act_hyper,
                            strides=stride,
                            padding=padding,
                            kernel_size=kernel_size,
                            kernel_regularizer=tf.keras.regularizers.l2(l2_hyper))(tensor)
    output_tensor=tensor
    #end output tensor
    print('output_tensor.shape',output_tensor.shape)
    model = Model(inputs=input_tensor,outputs=output_tensor)
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    loss_tf = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt,loss=loss_tf)
    print(model.summary())
    plot_model(model, to_file='unet_hyper.png', show_shapes=True, show_layer_names=True)
    return model
    
if __name__=='__main__':
    for i in range(5):
        print()

    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print('GPUS:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_visible_devices)

    print('building the hyperband tuner')
    max_epochs=50
    factor=3
    hyperband_iter=2
    
    print('max_epochs:',max_epochs)
    print('factor:',factor)
    print('hyper_iter:',hyperband_iter)
    
    tuner = kt.Hyperband(build_unet,
                        objective='val_loss',
                        max_epochs=max_epochs,
                        factor=factor,
                        hyperband_iterations=hyperband_iter,
                        directory='/scratch/bmac87/BC_hyper/',
                        project_name='UNet_factor_%s_hyper_iter_%s_max_epochs_%s'%(factor,hyperband_iter,max_epochs))
    
    print('tuner built successfully')
    print(tuner)

    print('loading the datasets for rotation 0')
    rot=0
    tfds_dir ='/scratch/bmac87/BC_tfds_v2/'
    train_tfds = tf.data.Dataset.load(tfds_dir+'rot_%s_train.tfds'%rot)
    bs = tf.data.experimental.cardinality(train_tfds).numpy()
    train_tfds = train_tfds.cache()
    train_tfds = train_tfds.shuffle(buffer_size=bs)
    train_tfds = train_tfds.batch(32)

    val_tfds = tf.data.Dataset.load(tfds_dir+'rot_%s_val.tfds'%rot)
    bs = tf.data.experimental.cardinality(val_tfds).numpy()
    val_tfds = val_tfds.cache()
    val_tfds = val_tfds.shuffle(buffer_size=bs)
    val_tfds = val_tfds.batch(32)

    # test_tfds = tf.data.Datasets.load(tfds_dir+'rot_%s_test.tfds'%rot)
    print('the datasets were loaded successfully')

    print('creating the early stopping call back')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    print('conducting the tuner.search')
    tuner.search(train_tfds, validation_data=val_tfds, callbacks=[stop_early])
    print('the search completed successfully')