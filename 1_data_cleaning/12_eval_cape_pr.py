import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pickle
import pandas as pd
import numpy as np

def main():
    print('evaluating the cape_pr model')
    init_time = '00Z'
    f_hour = 'f000'
    cpr_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/11_CAPE_PR_model/%s/%s/'%(init_time,f_hour)
    load_files = sorted(os.listdir(cpr_dir))
    for f,load_file in enumerate(load_files):
        if f==0:
            print(load_file)
            cpr_ds = xr.open_dataset(cpr_dir+load_file,engine='netcdf4')
            lat = cpr_ds['lat'].values
            lon = cpr_ds['lon'].values

            print(cpr_ds)
            valid_times = cpr_ds['valid_times'].values
            cpr_np = cpr_ds['cape_pr'].values

            print(cpr_np.shape)
            for i in range(cpr_np.shape[0]):

                ts = pd.Timestamp(valid_times[i])
                hr = ts.hour
                mo = ts.month
                yr = ts.year
                day = ts.day

                fig, axes = plt.subplots(figsize=(10,8),
                            nrows=1,
                            ncols=2,
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            layout='constrained')

                edgecolor='white'
                cb = axes[0].pcolormesh(lon,lat,cpr_np[i,:,:])
                axes[0].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
                axes[0].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
                axes[0].set_title('CAPE*PR')
                plt.colorbar(cb,ax=axes[0],label='CAPE*Precip_Rate')
                
                glm_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/1_GLM_downselect/CONUS/'
                glm_file = str(yr)+'.nc'
                glm_ds = xr.open_dataset(glm_dir+glm_file,engine='netcdf4')
                fed = glm_ds.sel(time=valid_times[i])['FED'].values
                binary = glm_ds.sel(time=valid_times[i])['binary_ltg'].values
                cb = axes[1].pcolormesh(lon,lat,fed)
                axes[1].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25)
                axes[1].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25)
                axes[1].set_title('GLM_FED')
                plt.colorbar(cb,ax=axes[1],label='GLM_FED')

                

                img_dir = './12_eval_cape_pr/'
                date_str = '%s_%s_%s_%sZ'%(f"{mo:02}",f"{day:02}",f"{yr:02}",f"{hr:02}")
                fname = load_file[:-10]+'.'+date_str+'.png'
                plt.savefig(img_dir+fname) 
                plt.close() 

                fig,axs = plt.subplots(figsize=(10,8),
                            nrows=1,
                            ncols=1)
                axs.scatter(np.ravel(fed),np.ravel(cpr_np[i,:,:]))
                axs.set_xlabel('GLM_FED')
                axs.set_ylabel('CAPE*PR')
                fname = load_file[:-10]+'.'+date_str+'.scatter.png'
                plt.savefig(img_dir+fname)
                plt.close()

                

if __name__=='__main__':
    main()