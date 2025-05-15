import tensorflow as tf
import pickle
import numpy as np
import xarray as xr
import argparse

def load_data_scratch(base_dir='/scratch/bmac87/BoltCast_scratch/',
                        rotation=0,
                        batch_size=16):
    
    train_file = 'rot_%s_train.tf'%(rotation)
    val_file = 'rot_%s_val.tf'%(rotation)
    test_file = 'rot_%s_test.tf'%(rotation)

    train_tf = tf.data.Dataset.load(base_dir+train_file)
    val_tf = tf.data.Dataset.load(base_dir+val_file)
    test_tf = tf.data.Dataset.load(base_dir+test_file)

    return train_tf, val_tf, test_tf

def min_max_scale(ds):

    data_np = ds['x'].values
    data_np2 = np.zeros(data_np.shape)
    for f,feature in enumerate(ds['features'].values):
        print('normalizing f,feature',f,feature)
        feature_data = np.squeeze(data_np[:,:,:,:,f])
        max_feature = np.max(np.max(np.max(np.max(feature_data,axis=3),axis=2),axis=1),axis=0)
        min_feature = np.min(np.min(np.min(np.min(feature_data,axis=3),axis=2),axis=1),axis=0)
        diff = max_feature-min_feature
        temp_data = (feature_data-min_feature)/diff
        data_np2[:,:,:,:,f] = temp_data
        del temp_data, feature_data, max_feature, min_feature
    del data_np
    return data_np2 

def min_max_scale_uno_datetime(data_np):
    data_np2 = np.zeros(data_np.shape)

    for f in range(9):
        feature_data = np.squeeze(data_np[:,:,:,f])
        max_feature = np.max(np.max(np.max(feature_data,axis=2),axis=1),axis=0)
        min_feature = np.min(np.min(np.min(feature_data,axis=2),axis=1),axis=0)
        diff = max_feature-min_feature
        temp_data = (feature_data-min_feature)/diff
        data_np2[:,:,:,f] = temp_data
        del temp_data, feature_data, max_feature, min_feature
    del data_np
    # print('data_np2.shape',data_np2.shape)
    return data_np2 

def build_rotations():
    print('building rotation file lists')
    rot_dict = {0:[],1:[],2:[],3:[],4:[]}

    for rot in rot_dict:
        if rot==0:
            train_files = ['summer_0.nc',
                            'fall_0.nc',
                            'winter_0.nc',
                            'spring_0.nc',
                            'summer_1.nc',
                            'fall_1.nc',
                            'winter_1.nc',
                            'spring_1.nc',
                            'summer_2.nc',
                            'fall_2.nc',
                            'winter_2.nc',
                            'spring_2.nc',
                            'summer_5.nc']
            
            val_files = ['summer_3.nc',
                            'fall_3.nc',
                            'winter_3.nc',
                            'spring_3.nc']

            test_files = ['summer_4.nc',
                            'fall_4.nc',
                            'winter_4.nc',
                            'spring_4.nc']

        elif rot==1:
            train_files = ['summer_1.nc',
                            'fall_1.nc',
                            'winter_1.nc',
                            'spring_1.nc',
                            'summer_2.nc',
                            'fall_2.nc',
                            'winter_2.nc',
                            'spring_2.nc',
                            'summer_3.nc',
                            'fall_3.nc',
                            'winter_3.nc',
                            'spring_3.nc',
                            'summer_5.nc']
            
            val_files = ['summer_4.nc',
                            'fall_4.nc',
                            'winter_4.nc',
                            'spring_4.nc']

            test_files = ['summer_0.nc',
                            'fall_0.nc',
                            'winter_0.nc',
                            'spring_0.nc']

        elif rot==2:
            train_files = ['summer_2.nc',
                            'fall_2.nc',
                            'winter_2.nc',
                            'spring_2.nc',
                            'summer_3.nc',
                            'fall_3.nc',
                            'winter_3.nc',
                            'spring_3.nc',
                            'summer_4.nc',
                            'fall_4.nc',
                            'winter_4.nc',
                            'spring_4.nc',
                            'summer_5.nc']
            
            val_files = ['summer_0.nc',
                            'fall_0.nc',
                            'winter_0.nc',
                            'spring_0.nc']

            test_files = ['summer_1.nc',
                            'fall_1.nc',
                            'winter_1.nc',
                            'spring_1.nc']

        elif rot==3:
            train_files = ['summer_3.nc',
                            'fall_3.nc',
                            'winter_3.nc',
                            'spring_3.nc',
                            'summer_4.nc',
                            'fall_4.nc',
                            'winter_4.nc',
                            'spring_4.nc',
                            'summer_0.nc',
                            'fall_0.nc',
                            'winter_0.nc',
                            'spring_0.nc',
                            'summer_5.nc']
            
            val_files = ['summer_1.nc',
                            'fall_1.nc',
                            'winter_1.nc',
                            'spring_1.nc']

            test_files = ['summer_2.nc',
                            'fall_2.nc',
                            'winter_2.nc',
                            'spring_2.nc']

        else:
            train_files = ['summer_4.nc',
                            'fall_4.nc',
                            'winter_4.nc',
                            'spring_4.nc',
                            'summer_0.nc',
                            'fall_0.nc',
                            'winter_0.nc',
                            'spring_0.nc',
                            'summer_1.nc',
                            'fall_1.nc',
                            'winter_1.nc',
                            'spring_1.nc',
                            'summer_5.nc']
            
            val_files = ['summer_2.nc',
                            'fall_2.nc',
                            'winter_2.nc',
                            'spring_2.nc']

            test_files = ['summer_3.nc',
                            'fall_3.nc',
                            'winter_3.nc',
                            'spring_3.nc']

        rot_dict[rot] = [train_files,val_files,test_files]
    return rot_dict

def load_test_data(base_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/binary/',
                    rotation=0,
                    batch_size=32):

    print('loading test data')
    rot_dict = build_rotations()
    rot_files = rot_dict[rotation]
    test_files = rot_files[2]
    data_list = []

    print(test_files)
    data_list = []
    for file in test_files:
        ds = xr.open_dataset(base_dir+file,engine='netcdf4')
        data_list.append(ds)
        del ds

    test_ds = xr.concat(data_list,dim='valid_times')
    lat = test_ds['lat'].values
    lon = test_ds['lon'].values
    days = test_ds['days'].values
    valid_times = test_ds['valid_times'].values
    features = test_ds['features'].values

    x = np.float32(min_max_scale(test_ds))
    print('x.shape',x.shape)
    y = np.float32(test_ds['y'].values)
    y = np.swapaxes(y,1,3)
    y = np.swapaxes(y,2,3)
    print('y.shape',y.shape)
    test_tf = tf.data.Dataset.from_tensor_slices((x,y))
    test_tf = test_tf.batch(batch_size)

    test_ds_norm = xr.Dataset(
                data_vars = dict(
                    
                    x = (["valid_times","days","lat","lon","features"],x),
                    y = (["valid_times","days","lat","lon"],y),
                ),

                coords = dict(
                    lon = (["lon"],lon),
                    lat = (["lat"],lat),
                    days = (["days"],days),
                    valid_times = (["valid_times"],valid_times),
                    features = (["features"],features)
                ),

                attrs = dict(
                    description="These x are normalized and the y are the binary lightning classification",
                )
            )
    return test_tf, test_ds_norm, test_ds

def load_test_data_nc(base_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/binary/',
                    rotation=0):
        
    print('loading test data')
    rot_dict = build_rotations()
    rot_files = rot_dict[rotation]
    test_files = rot_files[2]
    data_list = []

    print(test_files)
    data_list = []
    for file in test_files:
        ds = xr.open_dataset(base_dir+file,engine='netcdf4')
        data_list.append(ds)
        del ds

    test_ds = xr.concat(data_list,dim='valid_times')
    return test_ds

    

    