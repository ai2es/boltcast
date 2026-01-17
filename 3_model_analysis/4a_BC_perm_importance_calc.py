import xarray as xr
import numpy as np
import tensorflow as tf
import keras
import copy
import pickle
import sys
import argparse
import pandas as pd
import os
import wandb
import socket
from BC_analysis_data_loader import *

def extract_slurm_env(args):
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")
    print(args)

def parse_args():
    parser = argparse.ArgumentParser(description='BoltCast', fromfile_prefix_chars='@')
    parser.add_argument('--perm_num',type=int,default=100)
    parser.add_argument('--rotation',type=int,default=0)
    parser.add_argument('--lrate',type=float,default=0.000010000)
    parser.add_argument('--exp',type=int,default=0)
    parser.add_argument('--model_type',type=str,default='LSTM')
    args = parser.parse_args()
    return args

def get_model_rot(args):
    print('setting the rotation based on the experiment number')
    if args.exp<=49:
        args.model_type='UNet'
        if args.exp<=9:
            args.rotation=0
        if args.exp>=10 and args.exp<=19:
            args.rotation=1
        if args.exp>=20 and args.exp<=29:
            args.rotation=2
        if args.exp>=30 and args.exp<=39:
            args.rotation=3
        if args.exp>=40 and args.exp<=49:
            args.rotation=4

    if args.exp>=50:
        args.model_type='LSTM'
        if args.exp>=50 and args.exp<=59:
            args.rotation=0
        if args.exp>=60 and args.exp<=69:
            args.rotation=1
        if args.exp>=70 and args.exp<=79:
            args.rotation=2
        if args.exp>=80 and args.exp<=89:
            args.rotation=3
        if args.exp>=90 and args.exp<=99:
            args.rotation=4
    print(args)
    return args

def calc_perm_topkl(args):
    print('constructing the metrics list')
    opt = keras.optimizers.Adam(learning_rate=args.lrate, amsgrad=False)
    loss_tf = tf.keras.losses.BinaryCrossentropy()
    auc_pr_tf = tf.keras.metrics.AUC(name='auc_PR',curve='PR')
    all_metrics = [auc_pr_tf]
    
    print('loading the test dataset')
    load_dir = '/scratch/bmac87/BC_test_data_nc/'
    load_file = 'rot_%s_test.nc'%(args.rotation)
    test_ds = xr.open_dataset(load_dir+load_file,engine='netcdf4')
    X = min_max_scale(test_ds)
    X = np.float32(X)
    y_true = np.float32(test_ds['y'].values)
    y_true = np.swapaxes(y_true,1,3)
    y_true = np.swapaxes(y_true,2,3)
    features = test_ds['features'].values
    del test_ds
    
    print('generating the index matrix')
    index_matrix = np.zeros((X.shape[0],args.perm_num))
    for p in range(args.perm_num):
        index_matrix[:,p] = np.random.choice(np.arange(0,X.shape[0]),replace=False,size=X.shape[0])
    index_matrix = index_matrix.astype(int)

    model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AIES_reviews_v2/'
    model_file = 'model.keras'

    print('loading the model')
    if args.model_type=='UNet':
        exp_name = 'BC_UNet_rot_%s_lrate_0.000010000_/'%(args.rotation)
    else:
        exp_name = 'BC_LSTM_rot_%s_lrate_0.000010000_/'%(args.rotation)
    model = tf.keras.models.load_model(model_dir+exp_name+model_file)
    print(model)
    print('model loaded successfully')
    
    print('compiling the model')
    model.compile(optimizer=opt,loss=loss_tf,metrics=all_metrics)

    print('evaluating the model, un-permuted')
    tfds = tf.data.Dataset.from_tensor_slices((X, y_true))
    tfds = tfds.batch(32)
    tfds = tfds.cache()
    y_eval_dict = model.evaluate(tfds,return_dict=True,verbose=0)
    del tfds

    for f,feature in enumerate(features):
        print('permuting,',f,feature)
        PR_list = [y_eval_dict['auc_PR']]
        for p in range(args.perm_num):
            if p>=0:
                if p%10==0:
                    print('evaluating', p, ' of ',args.perm_num)

                #get shuffled indices along the samples
                if p==0:
                    X_shuffled = np.float32(copy.deepcopy(X))
                    X_shuffled[:,:,:,:,f] = X_shuffled[:,:,:,:,f][index_matrix[:,p]]
                else:
                    X_shuffled[:,:,:,:,f] = X_shuffled[:,:,:,:,f][index_matrix[:,p]]

                #generate the new tensorflow dataset then evaluate the shuffled dataset
                tfds_permuted = tf.data.Dataset.from_tensor_slices((X_shuffled, y_true))
                tfds_permuted = tfds_permuted.cache()
                tfds_permuted = tfds_permuted.batch(32)
                metrics_shuffled = model.evaluate(tfds_permuted,return_dict=True,verbose=1)
                del tfds_permuted

                #save off the metrics
                PR_list.append(metrics_shuffled['auc_PR'])
                del metrics_shuffled

        metrics_df = pd.DataFrame({'auc_PR':PR_list})#,'ROC':ROC_list,'binary_accuracy':acc_list,'precision':prec_list,'recall':rec_list})
        save_dir = '/scratch/bmac87/BoltCast_permutations/'
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        fsave = '%s_rot_%s_%s_perm_num_%s_exp_%s.pkl'%(feature,args.rotation,args.model_type,args.perm_num,args.exp)
        print('saving the permutation metrics to: ',save_dir+fsave)
        pickle.dump(metrics_df,open(save_dir+fsave,'wb'))
        del metrics_df 
        wandb.finish()

def start_wandb(args):

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
    return config_dict

def start_wandb(args,config_dict={}):
    #####
    # Start wandb
    wandb_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/wandb/'
    if os.path.isdir(wandb_dir)==False:
        os.makedirs(wandb_dir)
    run = wandb.init(dir=wandb_dir,
                    project='BC_perm_imp_v2_Dec2025', 
                    name='BC_rot_%s_%s_perm_num_%s_exp_%s.pkl'%(args.rotation,args.model_type,args.perm_num,args.exp), 
                    notes='BC_rot_%s_%s_perm_num_%s_exp_%s.pkl'%(args.rotation,args.model_type,args.perm_num,args.exp), 
                    config=config_dict)
    wandb.log({'hostname': socket.gethostname()})
    wandb.run.log_code(".")

if __name__=="__main__":
    print('BC_perm_importance.py')
    print('the modules loaded successfully')
    args = parse_args()
    print(args)
    extract_slurm_env(args=args)
    start_wandb(args=args)
    args = get_model_rot(args=args)#experiment control
    calc_perm_topkl(args=args)
    print('main method complete')