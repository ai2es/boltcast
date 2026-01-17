import os
import xarray as xr
import numpy as np
import shutil
import pickle
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

def count_flashes_per_rotation():
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

def glm_climo_calc():
    print('building the glm climo')
    rots = [0]
    data_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/fed/'
    rot_dict = build_rotations()
    data_types = ['train','val','test']
    ds_list = []
    for r,rot in enumerate(rots):
        for t,data_type in enumerate(data_types): 
            if r>=0:
                files = rot_dict[rot]
                data_files = files[t]
                for f,file in enumerate(data_files):
                    ds = xr.open_dataset(data_dir+file,engine='netcdf4')
                    ds_list.append(ds)
                    del ds
    total_ds = xr.concat(ds_list,dim='valid_times')
    total_ds = total_ds.sel(valid_times=total_ds['valid_times'].dt.hour == 0)
    print(total_ds['valid_times'])
    del ds_list
    y = total_ds['y'].values
    y = np.squeeze(y[:,:,:,0])#extract only the first day to not count the flashes more than once
    del total_ds
    print(y.shape)#samples,lat,lon
    y_sum = np.sum(y,axis=0)
    print(y_sum.shape)
    pickle.dump(y_sum,open('./glm_climo.pkl','wb'))
    del y_sum, y

def glm_climo_plot():
    print('plotting the glm climo')
    data = pickle.load(open('./glm_climo.pkl','rb'))
    print(data.shape)
    gfs_grid = pickle.load(open('./gfs_grid.pkl','rb'))
    lon = gfs_grid['lon']
    lat = gfs_grid['lat']

    data = data/(27*27)#(flashes per km^2 day)
    edgecolor='white'
    cmap='viridis'
    print(np.max(np.max(data)))
    vmin=0
    vmax=300
    fig, ax = plt.subplots(figsize=(20,18),nrows=1,ncols=1,subplot_kw={'projection': ccrs.PlateCarree()})
    im = ax.pcolormesh(lon,lat,data,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25,transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25,transform=ccrs.PlateCarree())
    cb = plt.colorbar(im,ax=ax, orientation='horizontal', pad=0.05)
    cb.set_label('Flash Extent Density (#/(Day km$^{2}$)', fontsize=24)
    cb.ax.tick_params(labelsize=20)
    plt.suptitle('GLM Climatology',fontsize=32)
    plt.tight_layout()
    plt.savefig('./AIES_results_reviewer_edits_v2/GLM_climo.png')
    plt.savefig('./AIES_results_reviewer_edits_v2/GLM_climo.pdf')
    plt.close()

if __name__=='__main__':
    glm_climo_plot()
    print('END OF 13_BC_rotation_flash_count.py')
