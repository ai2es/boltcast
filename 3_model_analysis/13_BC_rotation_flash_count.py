import os
import xarray as xr
import numpy as np
import shutil
import pickle
import glob

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

def main():
    print('counting the lightning flashes')
    rots = [0,1,2,3,4]
    data_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/fed/'
    rot_dict = build_rotations()
    data_types = ['train','val','test']

    for r,rot in enumerate(rots):
        for t,data_type in enumerate(data_types): 
            if r>=0:
                files = rot_dict[rot]
                data_files = files[t]
                ds_list = []
                for f,file in enumerate(data_files):
                    ds = xr.open_dataset(data_dir+file,engine='netcdf4')
                    if f==0 and r==0:
                        lat = ds['lat'].values
                        lon = ds['lon'].values
                        lon2d, lat2d = np.meshgrid(lon,lat)
                        g16_mask = lon2d>=260
                        g1718_mask = lon2d<260
                    ds_list.append(ds)
                    del ds
                train_ds = xr.concat(ds_list,dim='valid_times')
                del ds_list
                y = train_ds['y'].values
                del train_ds
                y_sum = np.squeeze(np.sum(y,axis=0))
                y_sum_1 = np.squeeze(y_sum[:,:,0])
                y_sum_g16 = np.sum(np.sum(y_sum_1*g16_mask))
                y_sum_g1718 = np.sum(np.sum(y_sum_1*g1718_mask))

                print('rotation: %s, %s'%(rot,data_type))
                print('G16 flashes:',y_sum_g16)
                print('G17/18 flashes:',y_sum_g1718)
                del y_sum, y_sum_1, y_sum_g16, y_sum_g1718

if __name__=='__main__':
    main()
    print('END OF 13_BC_rotation_flash_count.py')
