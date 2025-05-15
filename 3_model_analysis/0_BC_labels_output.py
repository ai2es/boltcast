import pickle
import matplotlib.pyplot as plt
import shutil 
import tensorflow as tf
from tensorflow import keras
import os
import xarray as xr
import numpy as np
import glob

def extract_slurm_env(args):
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")
    print(args)

def build_labels_outputs(model2pred='LSTM',rot=4):

    model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/models/'
    glob_call = 'BC_%s_rot_%s_*.keras'%(model2pred,rot)
    glob_call = model_dir+glob_call
    files = glob.glob(glob_call)
    print(files)

    print("loading the test data, tf and ds")
    data_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/tfds/'
    test_tf = tf.data.Dataset.load(data_dir+'rot_%s_test.tf'%(rot))
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
    
    for model_file in files:
        model = tf.keras.models.load_model(model_file)
        print("running the model.predict(test_tf)")
        model_output = model.predict(test_tf)

        dict_out = {'labels':np.float32(labels_all),
                        'model_output':np.float32(model_output)}

        print('saving off the output and labels')
        save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/labels_outputs/'
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        fsave = '%s_output_labels.pkl'%(model_file[72:-6])
        print(fsave)
        pickle.dump(dict_out,open(save_dir+fsave,'wb'))
        del dict_out, model_output

if __name__=="__main__":
    build_labels_outputs(model2pred='LSTM',rot=0)
    build_labels_outputs(model2pred='LSTM',rot=1)
    build_labels_outputs(model2pred='LSTM',rot=2)
    build_labels_outputs(model2pred='LSTM',rot=3)
    build_labels_outputs(model2pred='LSTM',rot=4)