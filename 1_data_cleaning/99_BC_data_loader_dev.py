import tensorflow as tf
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
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

def load_data(base_dir = '/scratch/bmac87/',
            rotation=0,
            batch_size=16):

    rot_dict = build_rotations()
    rot_files = rot_dict[rotation]
    
    for s,set_type in enumerate(['train','val','test']):
        
        files = rot_files[s]
        data_list = []
        if s==0:#training dataset
            print("building the training dataset")
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
            train_tf = train_tf.batch(batch_size)
            del data_list, x, y, train_ds

        elif s==1:
            print("building the validation dataset")
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
            val_tf = val_tf.batch(batch_size)
            del data_list,x,y,val_ds

        else:
            print("building the testing dataset")
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
            test_tf = test_tf.batch(batch_size)
            del data_list,x,y,test_ds
    return train_tf,val_tf,test_tf

def min_max_scale(ds):

    data_np = ds['x'].values
    for i in range(9):
        temp_data = np.squeeze(data_np[:,:,:,:,i])
        max_temp = np.max(np.max(np.max(np.max(temp_data,axis=3),axis=2),axis=1),axis=0)
        min_temp = np.min(np.min(np.min(np.min(temp_data,axis=3),axis=2),axis=1),axis=0)
        diff = max_temp-min_temp
        temp_data = (temp_data-min_temp)/diff
        data_np[:,:,:,:,i] = temp_data
        del temp_data
    return data_np 

def test_norm(rotation=0,
            base_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/'):

    rot_dict = build_rotations()
    rot_files = rot_dict[rotation]
    
    for s,set_type in enumerate(['train','val','test']):
        
        files = rot_files[s]
        data_list = []

        if s==0:#training dataset
            print("building the training dataset")
            for file in files:
                ds = xr.open_dataset(base_dir+file,engine='netcdf4')
                data_list.append(ds)
                del ds

            train_ds = xr.concat(data_list,dim='valid_times')
            print('normalizing')
            norm_train = min_max_scale(train_ds)
            print(norm_train.shape)
    out_np = train_ds['y'].values
    print(out_np.shape)

    for i in range(9):
        for t in range(1088):
            if t%250==0:

                fig, axes = plt.subplots(4, 2, figsize=(8, 6))
                axes[0,0].imshow(norm_train[t,0,:,:,i])
                # print(norm_train[t,0,:,:,i])
                axes[1,0].imshow(norm_train[t,1,:,:,i])
                # print(norm_train[t,1,:,:,i])
                axes[2,0].imshow(norm_train[t,2,:,:,i])
                # print(norm_train[t,2,:,:,i])
                axes[3,0].imshow(norm_train[t,3,:,:,i])
                # print(norm_train[t,3,:,:,i])

                axes[0,1].imshow(out_np[t,:,:,0])
                axes[1,1].imshow(out_np[t,:,:,1])
                axes[2,1].imshow(out_np[t,:,:,2])
                axes[3,1].imshow(out_np[t,:,:,3])

                plt.savefig('./test_norm_images/'+str(t)+'_'+str(i)+'.png')
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
    data_list = []

    print(test_files)
    data_list = []
    for file in test_files:
        ds = xr.open_dataset(base_dir+file,engine='netcdf4')
        data_list.append(ds)
        del ds
    test_ds = xr.concat(data_list,dim='valid_times')
    x = np.float32(min_max_scale(test_ds))
    y = np.float32(test_ds['y'].values)
    y = np.swapaxes(y,1,3)
    y = np.swapaxes(y,2,3)
    test_tf = tf.data.Dataset.from_tensor_slices((x,y))
    test_tf = test_tf.batch(batch_size)
    print(test_tf)

if __name__=='__main__':

    # visible_devices = tf.config.get_visible_devices('GPU') 
    # n_visible_devices = len(visible_devices)
    # print(n_visible_devices)
    # tf.config.set_visible_devices([], 'GPU')
    # print('NO VISIBLE DEVICES!!!!')
    
    parser = argparse.ArgumentParser(description='BoltCast', fromfile_prefix_chars='@')
    parser.add_argument('--rot', type=int,default=0)
    args = parser.parse_args()
    rot = args.rot
    print(rot)

    print('rotation: ',rot)
    load_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/binary/'
    train_tf, val_tf, test_tf = load_data(rotation=rot,
                                            batch_size=32,
                                            base_dir = load_dir)
    # print('training tf shape')
    # tfds=train_tf
    # c=0
    # for in_for,out_for in tfds:
    #     if c==0:
    #         inputs=in_for
    #     else:
    #         data = in_for
    #         inputs = np.concatenate([inputs,data],axis=0)
    #     c+=1
    # print(inputs.shape)

    # print('validation tf shape')
    # tfds=val_tf
    # c=0
    # for in_for,out_for in tfds:
    #     if c==0:
    #         inputs=in_for
    #     else:
    #         data = in_for
    #         inputs = np.concatenate([inputs,data],axis=0)
    #     c+=1
    # print(inputs.shape)

    # print('test tf shape')
    # tfds=test_tf
    # c=0
    # for in_for,out_for in tfds:
    #     if c==0:
    #         inputs=in_for
    #     else:
    #         data = in_for
    #         inputs = np.concatenate([inputs,data],axis=0)
    #     c+=1
    # print(inputs.shape)

    save_dir = '/scratch/bmac87/BoltCast_scratch/data/binary/'
    train_tf.save(save_dir+'rot_%s_train.tf'%(rot))
    val_tf.save(save_dir+'rot_%s_val.tf'%(rot))
    test_tf.save(save_dir+'rot_%s_test.tf'%(rot))
    del train_tf, val_tf, test_tf

    

    

    