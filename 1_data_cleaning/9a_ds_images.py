import numpy as np
import os
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

def main():

    init_time='18Z'
    ds_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/9_ds/'
    x_ds = xr.open_dataset(ds_dir+init_time+'_x.nc',engine='netcdf4')
    x_ds = x_ds['x']
    
    lat = x_ds['lat']
    lon = x_ds['lon']

    days = ['day_1','day_2','day_3','day_4']
    times = x_ds['valid_times'].values
    vars = ['cape','lifted_idx','reflectivity','precip_rate','w','ice_q','snow_q','graupel_q','rain_q']
    
    for var in vars:
        data = x_ds.sel(features=var)
        for day in days:
            data1 = data.sel(days=day)
            for t,time in enumerate(times):
                if t%250==0:
                    data2 = data1.sel(valid_times=time).values
                    fig,axs = plt.subplots(nrows=1,
                                    ncols=1, 
                                    subplot_kw={'projection': ccrs.PlateCarree()})

                    axs.add_feature(cfeature.COASTLINE,edgecolor="black")
                    axs.add_feature(cfeature.LAND)
                    axs.add_feature(cfeature.OCEAN)
                    axs.add_feature(cfeature.STATES,edgecolor="black")
                    cb0 = axs.pcolormesh(lon,lat,data2)
                    plt.suptitle(init_time+'_'+var+'_'+str(t))
                    plt.savefig('./9a_ds_images/'+init_time+'_'+var+'_'+str(t)+'.png')
                    plt.close()
        
if __name__=="__main__":

    main()