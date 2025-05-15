import sys
import argparse
import pickle
import pandas as pd
import wandb
import socket
import matplotlib.pyplot as plt
import shutil 
import os

import tensorflow as tf
from tensorflow import keras

#import BoltCast specific code
from BC_parser import *
from BC_unet import *
from BC_convLSTM import *
from BC_data_loader import * 

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

#################################################################
def check_args(args):
    '''
    Check that the input arguments are rational
    '''
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
#################################################################

def generate_fname(args):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.
    '''
    fname = 'BC_%s_rot_%s'%(args.model2train,args.rotation)

    label_str = args.label

    if args.model2train=='LSTM':
        conv_str = '_conv_%s'%args.lstm_conv_size
        conv_deep = '_conv_deep_%s'%args.lstm_conv_deep
        lstm_deep = '_lstm_deep_%s'%args.lstm_deep

    if args.model2train=='UNet':
        conv_str = '_conv_%s'%args.conv_size
        conv_deep = '_conv_deep_%s'%args.deep
        lstm_deep = ''

    # Put it all together, including #of training folds and the experiment rotation
    return fname+conv_str+conv_deep+lstm_deep+label_str

def execute_exp(args=None, multi_gpus=False):

    #Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch*multi_gpus

    print('Batch size', args.batch)

    ####################################################
    # Create the TF datasets for training, validation, testing

    if args.verbose >= 3:
        print('Starting data flow')

    if args.load_data:
        #load the data
        print('loading the data in BC_train.py')
        print('shuffle:',args.shuffle)

        ds_train, ds_val, ds_test = load_data_from_tfds(rotation=args.rotation, 
                                                base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/tfds/',
                                                batch_size=args.batch,
                                                shuffle=args.shuffle)

    ####################################################
    # Output file base and pkl file
    fbase = generate_fname(args)
    print(fbase)
    fname_out = "%s_results.pkl"%fbase
    print(fname_out)

    

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists"%fname_out)
        return

    #load the slurm environment variables into dictionary variable for documenting into wandb
    slurm_dict = {}
    slurm_dict['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
    slurm_dict['slurm_job_name'] = os.environ.get('SLURM_JOB_NAME')
    slurm_dict['slurm_job_account'] = os.environ.get('SLURM_JOB_ACCOUNT')
    slurm_dict['slurm_cpus_per_task'] = os.environ.get('SLURM_CPUS_PER_TASK')
    slurm_dict['slurm_nodelist'] = os.environ.get('SLURM_JOB_NODELIST')
    slurm_dict['slurm_partition'] = os.environ.get('SLURM_JOB_PARTITION')
    slurm_dict['slurm_num_nodes'] = os.environ.get('SLURM_JOB_NUM_NODES')
    slurm_dict['slurm_array_job_id'] = os.environ.get('SLURM_ARRAY_JOB_ID')
    slurm_dict['slurm_task_id'] = os.environ.get('SLURM_ARRAY_TASK_ID')

    #load the variables into dictionary for passing into wandb
    args_dict = vars(args)
    config_dict = {}
    for key in args_dict:
        config_dict[key] = args_dict[key]
    for key in slurm_dict:
        config_dict[key] = slurm_dict[key]

    #####
    # Start wandb
    wandb_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/wandb/'
    if os.path.isdir(wandb_dir)==False:
        os.makedirs(wandb_dir)
    run = wandb.init(dir=wandb_dir,
                    project=args.project, 
                    name=fbase, 
                    notes=fbase, 
                    config=config_dict)

    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Log the code
    wandb.run.log_code(".")

    #####
    # Callbacks
    cbs = []

    if args.early_stopping:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True,
                                                        min_delta=args.min_delta, monitor=args.monitor)
        cbs.append(early_stopping_cb)
    
    ckpt_dir = args.ckpt_path
    if os.path.isdir(ckpt_dir)==False:
        os.makedirs(ckpt_dir)
    ckpt_fname = fname_out[:-12]+'_checkpoint.model.keras'
    print('checkpoint info')
    print(ckpt_dir+ckpt_fname)
    cbs.append(tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir+ckpt_fname,
                                                    monitor='val_loss',
                                                    mode='auto',
                                                    save_best_only=False,
                                                    save_freq='epoch'))

    # Weights and Biases
    wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
    cbs.append(wandb_metrics_cb)

    if args.verbose >= 3:
        print('Fitting model')

    if multi_gpus<=1:
        if args.build_model:
            print('building the model')
            if args.model2train=='UNet':
                model = create_stacked_unet(args)
                print(model.summary())

            if args.model2train=='LSTM':
                print('creating the LSTM model')
                model = create_LSTM(args)
                print(model.summary())

            # Plot the model if the model is built
            if args.render:
                print('rendering the model figure')
                if os.path.isdir(args.results_path)==False:
                    os.makedirs(args.results_path)
                render_fname = args.results_path+'%s_model_plot.png'%fbase
                plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)
                wandb.log({'model architecture': wandb.Image(render_fname)})
            
            #declare the optimizer, loss function, 
            #and metric to track in wandb during training
            opt = keras.optimizers.Adam(learning_rate=args.lrate, amsgrad=False)
            loss_tf = tf.keras.losses.BinaryCrossentropy()
            metric_tf = tf.keras.metrics.BinaryAccuracy()
            model.compile(optimizer=opt,loss=loss_tf,metrics=metric_tf)
            if args.nogo:
                print("NO GO")
                return
    else:
        if args.build_model:
            # Starting parallelization
            print('starting parallelization: tf.distribute.MirroredStrategy')
            strategy = tf.distribute.MirroredStrategy()
            print(strategy)
            print('with strategy.scope()')
            with strategy.scope():
                
                print('building the model')
                if args.model2train=='UNet':
                    model = create_stacked_unet(args)
                    print(model.summary())
                if args.model2train=='LSTM':
                    model = create_LSTM(args)
                    print(model.summary())

                # Plot the model if the model is built
                if args.render:
                    if os.path.isdir(args.results_path)==False:
                        os.makedirs(args.results_path)
                    render_fname = args.results_path+'%s_model_plot.png'%fbase
                    plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)
                    wandb.log({'model architecture': wandb.Image(render_fname)})
                
                #declare the optimizer, loss function, 
                #and metric to track in wandb during training
                opt = keras.optimizers.Adam(learning_rate=args.lrate, amsgrad=False)
                loss_tf = tf.keras.losses.BinaryCrossentropy()
                metric_tf = tf.keras.metrics.BinaryAccuracy()
                model.compile(optimizer=opt,loss=loss_tf,metrics=metric_tf)
                if args.nogo:
                    print("NO GO")
                    return
    
    history = model.fit(ds_train,
                        batch_size = args.batch,
                        epochs=args.epochs,
                        use_multiprocessing=True, 
                        verbose=args.verbose>=2,
                        validation_data = ds_val,
                        callbacks=cbs)

    # Done training
    print('Done Training')
    # Generate results data
    results = {}
    results['history'] = history.history
    results['config']  = config_dict
    print('Predicting the model')
    
    # Save results
    fbase = generate_fname(args)
    results['fname_base'] = fbase
    if os.path.isdir(args.results_path)==False:
        os.makedirs(args.results_path)
    with open(args.results_path+"%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        print('saving the model')
        model_dir = args.results_path+'models/'
        if os.path.isdir(model_dir)==False:
            os.makedirs(model_dir)
        model.save(model_dir+"%s_model.keras"%(fbase))
    wandb.finish()

    return model

def main():

    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    #load the experiment infor
    print('loading the experiment for str(args.exp):',args.exp)
    if args.exp==1:
        args.rotation = 0
        args.lstm_deep = 3
        args.lstm_conv_deep = 2
    if args.exp==2:
        args.rotation = 2
        args.lstm_deep = 3
        args.lstm_conv_deep = 2
    if args.exp==3:
        args.rotation = 0
        args.lstm_deep = 2
        args.lstm_conv_deep = 2
    if args.exp==4:
        args.rotation = 1
        args.lstm_deep = 2
        args.lstm_conv_deep = 2
    if args.exp==5:
        args.rotation = 4
        args.lstm_deep = 2
        args.lstm_conv_deep = 2
    print(args.rotation,args.lstm_deep,args.lstm_conv_deep)

    # model_check = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/models/'
    # fcheck = 'BC_LSTM_rot_%s_conv_4_conv_deep_%s_lstm_deep_%s_no_drop_no_shuffle_model.keras'%(args.rotation,args.lstm_conv_deep,args.lstm_deep)
    # print('checking:',fcheck)
    # if os.path.isfile(model_check+fcheck):
    #     print('already trained:',fcheck)
    #     return

    if args.verbose >= 3:
        print('Arguments parsed')

    #GPU check
    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print("number of  GPUs: ",n_visible_devices )
    print('GPU info:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_visible_devices)
    else:
        print('NO GPU')

    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        visible_devices = tf.config.get_visible_devices('GPU') 
        n_visible_devices = len(visible_devices)
        tf.config.set_visible_devices([], 'GPU')
        print('GPUs turned off')
    print()

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
    execute_exp(args, multi_gpus=n_visible_devices)

if __name__ == "__main__":
    main()