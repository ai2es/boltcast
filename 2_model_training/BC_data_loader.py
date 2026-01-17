import tensorflow as tf
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse
import os

def load_data_from_tfds(base_dir='/scratch/bmac87/BoltCast_scratch/',
                        rotation=0,
                        batch_size=32):
    
    train_file = 'rot_%s_train.tfds'%(rotation)
    val_file = 'rot_%s_val.tfds'%(rotation)
    test_file = 'rot_%s_test.tfds'%(rotation)

    train_tf = tf.data.Dataset.load(base_dir+train_file)
    bs = tf.data.experimental.cardinality(train_tf).numpy()
    train_tf = train_tf.cache()
    train_tf = train_tf.shuffle(buffer_size=bs)
    train_tf = train_tf.batch(batch_size)

    val_tf = tf.data.Dataset.load(base_dir+val_file)
    bs = tf.data.experimental.cardinality(val_tf).numpy()
    val_tf = val_tf.cache()
    val_tf = val_tf.shuffle(buffer_size=bs)
    val_tf = val_tf.batch(batch_size)

    test_tf = tf.data.Dataset.load(base_dir+test_file)
    bs = tf.data.experimental.cardinality(test_tf).numpy()
    test_tf = test_tf.cache()
    test_tf = test_tf.shuffle(buffer_size=bs)
    test_tf = test_tf.batch(batch_size)
    return train_tf, val_tf, test_tf

def cconcatenate_global(base_dir = '/scratch/bmac87/',
            rotation=0):

    rot_dict = build_rotations()
    rot_files = rot_dict[rotation]
    ds_list = []
    for s,set_type in enumerate(['train','val','test']):
        files = rot_files[s]
        for f,file in enumerate(files):
            ds = xr.open_dataset(base_dir+file,engine='netcdf4')
            if f==0 and s==0:
                features = ds['features'].values
            ds_list.append(ds)
    ds_global = xr.concat(ds_list,dim='valid_times')    
    ds_global.to_netcdf('/scratch/bmac87/minmax_global.nc',engine='netcdf4')
    del ds_global

def calc_global_min_max(base_dir='/scratch/bmac87/'):
    ds = xr.open_dataset(base_dir+'minmax_global.nc',engine='netcdf4')
    x = ds['x'].values
    features = ds['features'].values
    maxes = np.max(np.max(np.max(np.max(x,axis=3),axis=2),axis=1),axis=0)
    print(maxes)
    mins = np.min(np.min(np.min(np.min(x,axis=3),axis=2),axis=1),axis=0)
    print(mins)
    minmax_dict = {'mins':mins,'maxes':maxes,'features':features}
    pickle.dump(minmax_dict,open('minmax_dict.pkl','wb'))

def load_data_from_seasons_nc(base_dir = '/scratch/bmac87/',
            rotation=0):

    rot_dict = build_rotations()
    rot_files = rot_dict[rotation]
    
    for s,set_type in enumerate(['train','val','test']):
        files = rot_files[s]
        data_list = []
        if s==0:#training dataset
            print("building the training dataset, rotation:",rot)
            for file in files:
                ds = xr.open_dataset(base_dir+file,engine='netcdf4')
                data_list.append(ds)
                del ds
            train_ds = xr.concat(data_list,dim='valid_times')
            x = np.float32(min_max_scale(train_ds))
            y = np.float32(train_ds['y'].values)
            y = np.swapaxes(y,1,3)
            y = np.swapaxes(y,2,3)
            train_tf = tf.data.Dataset.from_tensor_slices((x,y))
            del data_list, x, y, train_ds
            print(train_tf)

        elif s==1:
            print("building the validation dataset, rotation:",rot)
            for file in files:
                ds = xr.open_dataset(base_dir+file,engine='netcdf4')
                data_list.append(ds)
                del ds
            val_ds = xr.concat(data_list,dim='valid_times')
            x = np.float32(min_max_scale(val_ds))
            y = np.float32(val_ds['y'].values)
            y = np.swapaxes(y,1,3)
            y = np.swapaxes(y,2,3)
            val_tf = tf.data.Dataset.from_tensor_slices((x,y))
            del data_list,x,y,val_ds
            print(val_tf)

        else:
            print("building the testing dataset, rotation:",rot)
            for file in files:
                ds = xr.open_dataset(base_dir+file,engine='netcdf4')
                data_list.append(ds)
                del ds
            test_ds = xr.concat(data_list,dim='valid_times')
            x = np.float32(min_max_scale(test_ds))
            y = np.float32(test_ds['y'].values)
            y = np.swapaxes(y,1,3)
            y = np.swapaxes(y,2,3)
            test_tf = tf.data.Dataset.from_tensor_slices((x,y))
            del data_list,x,y,test_ds
            print(test_tf)
    return train_tf,val_tf,test_tf

def min_max_scale(ds):
    minmax_dict = pickle.load(open('minmax_dict.pkl','rb'))
    maxes = minmax_dict['maxes']
    mins = minmax_dict['mins']
    features = minmax_dict['features']
    data_np = ds['x'].values
    for f,feature in enumerate(features):
        temp_data = np.squeeze(data_np[:,:,:,:,f])
        max_temp = maxes[f]
        min_temp = mins[f]
        diff = max_temp-min_temp
        temp_data = (temp_data-min_temp)/diff
        data_np[:,:,:,:,f] = temp_data
        del temp_data
    return data_np 

def test_norm(rotation=0,
            base_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/'):

    rot_dict = build_rotations()
    rot_files = rot_dict[rotation]

    for s,set_type in enumerate(['train','val','test']):
        files = rot_files[s]
        data_list = []
        if s==2:#test dataset
            print("building the test dataset")
            for file in files:
                ds = xr.open_dataset(base_dir+file,engine='netcdf4')
                data_list.append(ds)
                del ds
            
            test_ds = xr.concat(data_list,dim='valid_times')
            features = test_ds['features'].values
            print('normalizing')
            norm_test = min_max_scale(test_ds)
            print(norm_test.shape)
    out_np = test_ds['y'].values
    print(out_np.shape)

    for f,feature in enumerate(features):
        for t in range(norm_test.shape[0]):#samples
            if t%50==0:
                fig, axes = plt.subplots(4, 2, figsize=(8, 6))
                
                im = axes[0,0].imshow(norm_test[t,0,:,:,f])
                plt.colorbar(im,ax=axes[0,0])
                
                axes[1,0].imshow(norm_test[t,1,:,:,f])
                plt.colorbar(im,ax=axes[1,0])
                
                axes[2,0].imshow(norm_test[t,2,:,:,f])
                plt.colorbar(im,ax=axes[2,0])

                axes[3,0].imshow(norm_test[t,3,:,:,f])
                plt.colorbar(im,ax=axes[3,0])

                axes[0,1].imshow(out_np[t,:,:,0])
                axes[1,1].imshow(out_np[t,:,:,1])
                axes[2,1].imshow(out_np[t,:,:,2])
                axes[3,1].imshow(out_np[t,:,:,3])

                plt.savefig('./test_norm_images/'+str(t)+'_'+feature+'.png')
                plt.close()

def build_rotations():

    rot_dict = {0:[],1:[],2:[],3:[],4:[]}

    for rot in rot_dict:
        print(rot)
        if rot==0:
            print('building rotation 0')
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
            print('building rotation 1')
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
            print('building rotation 2')
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
            print('building rotation 3')
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
            print('building rotation 4')

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

def load_test_data(base_dir = '/scratch/bmac87/',
                    rotation=0,
                    batch_size=16):

    print('loading test data')
    rot_dict = build_rotations()
    rot_files = rot_dict[rotation]
    test_files = rot_files[2]
    print(test_files)
    data_list = []

    print(test_files)
    data_list = []
    for file in test_files:
        ds = xr.open_dataset(base_dir+file,engine='netcdf4')
        data_list.append(ds)
        del ds
    test_ds = xr.concat(data_list,dim='valid_times')
    save_dir = '/scratch/bmac87/BC_test_data_nc/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    test_ds.to_netcdf(save_dir+'rot_%s_test.nc'%rotation,engine='netcdf4')

    # x = np.float32(min_max_scale(test_ds))
    # y = np.float32(test_ds['y'].values)
    # y = np.swapaxes(y,1,3)
    # y = np.swapaxes(y,2,3)
    # test_tf = tf.data.Dataset.from_tensor_slices((x,y))
    # test_tf = test_tf.batch(batch_size)
    # print(test_tf)

if __name__=='__main__':
    print('in BC_data_loader.py main function')
    base_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/binary/'
    load_test_data(base_dir=base_dir,
                    rotation=0)
    load_test_data(base_dir=base_dir,
                    rotation=1)
    load_test_data(base_dir=base_dir,
                    rotation=2)
    load_test_data(base_dir=base_dir,
                    rotation=3)
    load_test_data(base_dir=base_dir,
                    rotation=4)
