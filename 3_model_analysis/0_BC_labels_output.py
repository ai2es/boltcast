import pickle
import matplotlib.pyplot as plt
import shutil 
import tensorflow as tf
from tensorflow import keras
import os
import xarray as xr
import numpy as np
import glob
import time

def extract_slurm_env(args):
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")
    print(args)

def build_labels_outputs(model_type='UNet',lrate='0.000010000',rot=0):
    exp_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AIES_reviews_v2/'
    rot_dir = 'BC_%s_rot_%s_lrate_%s_/'%(model_type,rot,lrate)
    model_dir = exp_dir+rot_dir
    model_file = 'model.keras'

    print("loading the test data, tf and ds")
    data_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/BC_tfds_v2/'
    test_tf = tf.data.Dataset.load(data_dir+'rot_%s_test.tfds'%(rot))
    test_tf = test_tf.batch(32)

    c=0
    print("loading the test_tf.take()")
    for inputs,labels in test_tf:
        if c==0:
            inputs_all = inputs
            labels_all = labels
        else:
            inputs_all = np.concatenate([inputs_all,inputs],axis=0)
            labels_all = np.concatenate([labels_all,labels],axis=0)
        c+=1
    del inputs_all
    
    model = tf.keras.models.load_model(model_dir+model_file)
    print("running the model.predict(test_tf)")
    start_clock = time.time()
    model_output = model.predict(test_tf)
    end_clock = time.time()
    print(model_type,lrate,rot,' test ds inference time:',(end_clock-start_clock)/60,'minutes')

    dict_out = {'labels':np.float32(labels_all),
                    'model_output':np.float32(model_output)}

    print('saving off the output and labels')
    save_dir = model_dir
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    fsave = 'output_labels.pkl'
    pickle.dump(dict_out,open(save_dir+fsave,'wb'))
    del dict_out, model_output

if __name__=="__main__":

    models = ['UNet','LSTM']
    lrates = ['0.000010000','0.000001000']
    rots = [0,1,2,3,4]

    for model in models:
        for lrate in lrates:
            for rot in rots:
                build_labels_outputs(rot=rot,model_type=model,lrate=lrate)