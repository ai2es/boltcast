import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pickle
import pandas as pd

mpl.rcParams['axes.labelsize'] = 18 #fontsize in pts
mpl.rcParams['axes.titlesize'] = 18 
mpl.rcParams['xtick.labelsize'] = 12 
mpl.rcParams['ytick.labelsize'] = 12 
mpl.rcParams['legend.fontsize'] = 18 

def get_outputs(rotation=4, 
                conv_deep=0,
                lstm_deep=1):

    print('extracting the unet outputs')
    output_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/UNet/labels_outputs/'
    unet_output_file = 'UNet_symmetric_rot_%s_output.pkl'%(rotation)
    unet_output = pickle.load(open(output_dir+unet_output_file,'rb'))
    y_pred_unet = unet_output['model_output']

    output_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/labels_outputs/'
    lstm_output_file = 'BC_LSTM_rot_%s_conv_4_conv_deep_%s_lstm_deep_%s_no_drop_no_shuffle_model_output_labels.pkl'%(rotation,conv_deep,lstm_deep)
    lstm_output = pickle.load(open(output_dir+lstm_output_file,'rb'))
    y_pred_lstm = lstm_output['model_output']
    return y_pred_unet, y_pred_lstm

def get_test_ds(rotation=0):
    print('get_test_ds, rotation: ',rotation)
    ds_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/tfds_binary_32batch/'
    ds_file = 'rot_%s_test.nc'%(rotation)
    test_ds = xr.open_dataset(ds_dir+ds_file,engine='netcdf4')
    lat = test_ds['lat']
    lon = test_ds['lon']
    valid_times = test_ds['valid_times'].values
    return test_ds, lat, lon, valid_times

def get_GFS_30dbz(init_time='00Z',date_np64='06/17/2023 00:00'):
    print('get_GFS_30dbz()')
    load_file = init_time+'_x.nc'
    load_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/9_ds/binary_ltg/'
    ds = xr.open_dataset(load_dir+load_file,engine='netcdf4')
    valid_times = ds['valid_times'].values
    x = ds['x'].values
    reflectivity = x[:,:,:,:,2] #index 2 is reflectivity, index 1 is cape
    cape = x[:,:,:,:,1]
    del ds, x
    ts = pd.Timestamp(date_np64)
    time_idx = np.where(valid_times==ts)[0]
    single_refl_output = np.squeeze(reflectivity[time_idx,:,:,:])
    single_refl_output = np.where(single_refl_output<30,np.nan,single_refl_output)
    single_refl_output = np.where(single_refl_output>=30,1,single_refl_output)

    single_cape_output = np.squeeze(cape[time_idx,:,:,:])
    return single_refl_output,single_cape_output

def generate_variable_model_output(rotation=0,
                                    lstm_deep=1,
                                    conv_deep=0):
    
    #valid_times is a list of the model run initialization times
    test_ds, lat, lon, valid_times = get_test_ds(rotation=rotation)
    
    for q in range(len(valid_times)):
        if q>=0:
            vt = valid_times[q]
            timestamp = pd.Timestamp(vt)
            hr = timestamp.hour
            hr_str = f"{hr:02}"
            day = timestamp.day
            day_str = f"{day:02}"
            yr = timestamp.year
            yr_str = f"{yr:02}"
            mo = timestamp.month
            mo_str = f"{mo:02}"

            print(hr_str, day_str, yr_str, mo_str)
            init_time = hr_str+'Z'

            dt = np.timedelta64(24,'h')
            eval_times = []
            eval_idx = []
            refl_list = []
            cape_list = []
            try:
                for v in range(4):
                    print("finding the reflectivities and time indexes,",vt)
                    refl_temp, cape_temp = get_GFS_30dbz(init_time=init_time,date_np64=vt)
                    refl_list.append(refl_temp)
                    cape_list.append(cape_temp)

                    idx = np.where(valid_times==vt)[0]
                    print(idx,vt)
                    eval_idx.append(idx)
                    eval_times.append(vt)
                    vt = vt+dt
                
                refl_4 = refl_list[0]
                refl_4 = refl_4[3,:,:]
                refl_3 = refl_list[1]
                refl_3 = refl_3[2,:,:]
                refl_2 = refl_list[2]
                refl_2 = refl_2[1,:,:]
                refl_1 = refl_list[3]
                refl_1 = refl_1[0,:,:]
            except Exception as e:
                print(e)
                print('no reflectivity')
                continue
            
            try:
                print('getting the ltg labels for: ',eval_times[3])
                print('test_ds[y].values.shape',test_ds['y'].values.shape)
                y_true = test_ds.sel(valid_times=eval_times[3])['y'].values[0,:,:]
                y_true = np.where(y_true<=.05,np.nan,y_true)
                print('y_true.shape',y_true.shape,' for ',eval_times[3])

                y_pred_unet, y_pred_lstm = get_outputs(rotation=rotation, 
                                                        conv_deep=conv_deep,
                                                        lstm_deep=lstm_deep)
                y_pred_unet = np.where(y_pred_unet <= 0.05, np.nan, y_pred_unet)
                y_pred_lstm = np.where(y_pred_lstm <= 0.05, np.nan, y_pred_lstm)
                print('y_pred_unet.shape',y_pred_unet.shape)
                print('type(y_pred_unet)',type(y_pred_unet))
            except Exception as e:
                print(e)
                print('no Unet data')
                continue
            
            print('getting day 4 unet prediction from ', eval_times[0], eval_idx[0])
            y_pred_unet_4 = np.squeeze(y_pred_unet[eval_idx[0],:,:,:,0])
            y_pred_unet_4 = y_pred_unet_4[3,:,:]#day 4

            print('getting day 3 unet prediction from ', eval_times[1],eval_idx[1])
            y_pred_unet_3 = np.squeeze(y_pred_unet[eval_idx[1],:,:,:,0])
            y_pred_unet_3 = y_pred_unet_3[2,:,:]#day 3
            
            print('getting day 2 unet prediction from ',eval_times[2],eval_idx[2])
            y_pred_unet_2 = np.squeeze(y_pred_unet[eval_idx[2],:,:,:,0])
            y_pred_unet_2 = y_pred_unet_2[1,:,:]#day 2

            print('getting day 1 unet prediction from ',eval_times[3],eval_idx[3])
            y_pred_unet_1 = np.squeeze(y_pred_unet[eval_idx[3],:,:,:,0])
            y_pred_unet_1 = y_pred_unet_1[0,:,:]#day 1

            print('getting day 4 lstm prediction from ',eval_times[0],eval_idx[0])
            y_pred_lstm_4 = np.squeeze(y_pred_lstm[eval_idx[0],:,:,:,0])
            y_pred_lstm_4 = y_pred_lstm_4[3,:,:]

            print('getting day 3 lstm prediction from ',eval_times[1],eval_idx[1])
            y_pred_lstm_3 = np.squeeze(y_pred_lstm[eval_idx[1],:,:,:,0])
            y_pred_lstm_3 = y_pred_lstm_3[2,:,:]

            print('getting day 2 lstm prediction from ',eval_times[2],eval_idx[2])
            y_pred_lstm_2 = np.squeeze(y_pred_lstm[eval_idx[2],:,:,:,0])
            y_pred_lstm_2 = y_pred_lstm_2[1,:,:]

            print('getting day 1 lstm prediction from ',eval_times[3],eval_idx[3])
            y_pred_lstm_1 = np.squeeze(y_pred_lstm[eval_idx[3],:,:,:,0])
            y_pred_lstm_1 = y_pred_lstm_1[0,:,:]

            print('generating the variable model figure')
            edgecolor='black'
            
            # Define the color segments and corresponding values
            colors = ["gray", "slateblue", "blue", "darkgreen", "green", "lightgreen", "yellow", "peru", "brown"]
            bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

            # Create a colormap and norm
            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # Set tick labels
            # cb.set_ticks(bounds[:-1])
            # cb.ax.set_xticklabels([str(b) for b in bounds[:-1]])
            vmin=10
            vmax=100
            fig, axes = plt.subplots(figsize=(20,8),
                                    nrows=3,
                                    ncols=5,
                                    subplot_kw={'projection': ccrs.PlateCarree()},
                                    layout='constrained')
            
            cb = axes[0,0].contourf(lon,lat,y_pred_unet_4*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[0,0].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[0,0].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            axes[0,0].set_xticks([])
            axes[0,0].set_yticks([])
            str_max = f"{np.nanmax(y_pred_unet_4)*100:.2f}"+'%'
            axes[0,0].set_xlabel('Max: %s'%(str_max))
            axes[0,0].set_ylabel('UNet',fontsize=18)
            axes[0,0].set_title('Day 4',fontsize=18)

            cb = axes[0,1].contourf(lon,lat,y_pred_unet_3*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[0,1].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[0,1].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            axes[0,1].set_xticks([])
            axes[0,1].set_yticks([])
            str_max = f"{np.nanmax(y_pred_unet_3)*100:.2f}"+'%'
            axes[0,1].set_xlabel('Max: %s'%(str_max))
            axes[0,1].set_title('Day 3',fontsize=18)

            cb = axes[0,2].contourf(lon,lat,y_pred_unet_2*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[0,2].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[0,2].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            axes[0,2].set_xticks([])
            axes[0,2].set_yticks([])
            str_max = f"{np.nanmax(y_pred_unet_2)*100:.2f}"+'%'
            axes[0,2].set_xlabel('Max: %s'%(str_max))
            axes[0,2].set_title('Day 2',fontsize=18)

            cb = axes[0,3].contourf(lon,lat,y_pred_unet_1*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[0,3].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[0,3].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            axes[0,3].set_xticks([])
            axes[0,3].set_yticks([])
            str_max = f"{np.nanmax(y_pred_unet_1)*100:.2f}"+'%'
            axes[0,3].set_xlabel('Max: %s'%(str_max))
            axes[0,3].set_title('Day 1',fontsize=18)

            cb = axes[0,4].contourf(lon,lat,y_true*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[0,4].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[0,4].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            axes[0,4].set_xticks([])
            axes[0,4].set_yticks([])
            axes[0,4].set_title('Labels',fontsize=18)

            cb = axes[1,0].contourf(lon,lat,y_pred_lstm_4*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[1,0].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[1,0].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            str_max = f"{np.nanmax(y_pred_lstm_4)*100:.2f}"+'%'
            axes[1,0].set_xlabel('Max: %s'%(str_max))
            axes[1,0].set_xticks([])
            axes[1,0].set_yticks([])
            axes[1,0].set_ylabel('LSTM',fontsize=18)

            cb = axes[1,1].contourf(lon,lat,y_pred_lstm_3*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[1,1].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[1,1].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            str_max = f"{np.nanmax(y_pred_lstm_3)*100:.2f}"+'%'
            axes[1,1].set_xlabel('Max: %s'%(str_max))
            axes[1,1].set_xticks([])
            axes[1,1].set_yticks([])

            cb = axes[1,2].contourf(lon,lat,y_pred_lstm_2*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[1,2].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[1,2].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            str_max = f"{np.nanmax(y_pred_lstm_2)*100:.2f}"+'%'
            axes[1,2].set_xlabel('Max: %s'%(str_max))
            axes[1,2].set_xticks([])
            axes[1,2].set_yticks([])

            cb = axes[1,3].contourf(lon,lat,y_pred_lstm_1*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[1,3].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[1,3].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            str_max = f"{np.nanmax(y_pred_lstm_1)*100:.2f}"+'%'
            axes[1,3].set_xlabel('Max: %s'%(str_max))
            axes[1,3].set_xticks([])
            axes[1,3].set_yticks([])
            axes[1,4].axis('off')

            cb = axes[2,0].pcolormesh(lon,lat,refl_4*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[2,0].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[2,0].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            axes[2,0].set_xticks([])
            axes[2,0].set_yticks([])
            axes[2,0].set_ylabel('Z >= 30dBZ')

            cb = axes[2,1].pcolormesh(lon,lat,refl_3*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[2,1].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[2,1].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            axes[2,1].set_xticks([])
            axes[2,1].set_yticks([])

            cb = axes[2,2].pcolormesh(lon,lat,refl_2*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[2,2].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[2,2].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            axes[2,2].set_xticks([])
            axes[2,2].set_yticks([])

            cb = axes[2,3].pcolormesh(lon,lat,refl_1*100,cmap=cmap,vmin=vmin,vmax=vmax)
            axes[2,3].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
            axes[2,3].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
            axes[2,3].set_xticks([])
            axes[2,3].set_yticks([])

            axes[2,4].axis('off')

            ts = pd.Timestamp(eval_times[3])
            yr = f"{ts.year:04}"
            mo = f"{ts.month:02}"
            day = f"{ts.day:02}"
            hr = f"{ts.hour:02}"

            date_str = '%s_%s_%s_%sZ'%(mo,day,yr,hr)
            date_str1 = '%s/%s/%s %sZ'%(mo,day,yr,hr)

            ts2 = pd.Timestamp(vt)
            yr = f"{ts2.year:04}"
            mo = f"{ts2.month:02}"
            day = f"{ts2.day:02}"
            hr = f"{ts2.hour:02}"
            date_str2 = '%s/%s/%s %sZ'%(mo,day,yr,hr)

            title_str = 'Lightning Valid Time: %s - %s'%(date_str1,date_str2)
            plt.suptitle(title_str,fontsize=24)
            
            save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/BoltCast_output_movie_pngs/%s_conv_deep_%s_lstm_deep/'%(conv_deep,lstm_deep)
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            save_file = 'variable_model_output_%s_rot_%s_conv_deep_%s_lstm_deep_%s.png'%(date_str,rotation,conv_deep,lstm_deep)
            plt.savefig(save_dir+save_file)
            plt.close()

def main():
    
    generate_variable_model_output(rotation=4,
                                    lstm_deep=1,
                                    conv_deep=0)

    # generate_variable_model_output(rotation=4,
    #                                 lstm_deep=1,
    #                                 conv_deep=1)
    
    # generate_variable_model_output(rotation=4,
    #                                 lstm_deep=1,
    #                                 conv_deep=2)


if __name__ == "__main__":
    main()
