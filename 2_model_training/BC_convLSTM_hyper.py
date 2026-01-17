import argparse
from keras.layers import Input, Conv2D, Conv2DTranspose, Conv3D, ConvLSTM2D, MaxPooling2D, MaxPooling3D, UpSampling3D, TimeDistributed, Concatenate, SpatialDropout3D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import keras
import numpy as np
from BC_parser import *
import keras_tuner as kt

def create_LSTM(hp):
    print("creating the LSTM in BC_convLSTM.py")
    image_size = [4,128,256,9]
    padding='same'
    stride = 1
    lrate=.00001
    filter_list = [32,64,128]
    num_filters = 16
    act_hyper = hp.Choice("activation",['elu','relu'])
    l2_hyper = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5, 0.0])
    dropout_hyper = hp.Float("dropout",min_value=0.0,max_value=0.1,step=0.05)
    kernel_hyper = hp.Choice("kernel_2d", [2, 3, 4])
    activation_last = 'sigmoid'
    ret_seq=True
    ret_state=True

    input_tensor = Input(shape=(image_size[0],image_size[1],image_size[2],image_size[3]),
                                    dtype=tf.dtypes.float32,
                                    name='input')
    tensor = input_tensor

    kernel_time = tensor.shape[1]
    kernel_size = (kernel_time,kernel_hyper,kernel_hyper)
    tensor = Conv3D(filters=num_filters,
                        input_shape = tensor.shape,
                        activation=act_hyper,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_hyper),
                        kernel_size = kernel_size,
                        name='encoder_conv_3d')(tensor)
    tensor = SpatialDropout3D(rate=dropout_hyper,
                                dtype=tf.dtypes.float32,
                                name='drop_encoder')(tensor)
    
    ##########################################CONV_LSTM_Layers#######################################################
    print("building the ConvLSTM layer")
    kernel_size = (kernel_hyper,kernel_hyper)
    tensor, h_tensor, c_tensor = ConvLSTM2D(filters=num_filters, 
                            kernel_size=kernel_size, 
                            padding=padding, 
                            return_sequences=ret_seq,
                            return_state=ret_state,
                            kernel_regularizer=tf.keras.regularizers.l2(l2_hyper),
                            input_shape = tensor.shape,
                            name='clstm_1')(tensor)
    ######################################################################################################################
    
    kernel_time = tensor.shape[1]
    kernel_size = (kernel_time,kernel_hyper,kernel_hyper)
    tensor = Conv3D(filters=num_filters,
                        input_shape=tensor.shape,
                        activation=act_hyper,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_hyper),
                        kernel_size=kernel_size,
                        name='decoder')(tensor)
    tensor = SpatialDropout3D(rate=dropout_hyper,
                                dtype=tf.dtypes.float32,
                                name='drop_decoder')(tensor)

    kernel_time = tensor.shape[1]
    kernel_size = (kernel_time,kernel_hyper,kernel_hyper)
    output_tensor = Conv3D(filters=1,
                        input_shape=tensor.shape,
                        activation=activation_last,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_hyper),
                        kernel_size=kernel_size,
                        name='output_layer')(tensor)

    #complete the model
    model = Model(inputs=input_tensor,outputs=output_tensor)
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    loss_tf = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt,loss=loss_tf)
    print(model.summary())
    plot_model(model, to_file='lstm_hyper.png', show_shapes=True, show_layer_names=True)
    return model

if __name__ == "__main__":
    
    print('BC_convLSTM.py main function')
    max_epochs=50
    factor=3
    hyperband_iter=2
    print('max_epochs:',max_epochs)
    print('factor:',factor)
    print('hyperband_iter:',hyperband_iter)
    
    tuner = kt.Hyperband(create_LSTM,
                        objective='val_loss',
                        max_epochs=max_epochs,
                        factor=factor,
                        hyperband_iterations=hyperband_iter,
                        directory='/scratch/bmac87/BC_hyper/',
                        project_name='LSTM_factor_%s_hyper_iter_%s_max_epochs_%s'%(factor,hyperband_iter,max_epochs))

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