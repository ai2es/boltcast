import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import shutil
import pandas as pd

def main():

    print("calculating cape * pr model")
    
    hours = []#int
    for i in range(33):
        hours.append(i*3)

    f_hours = []#string
    for hr in hours:
        f_hours.append('f'+f"{hr:03}")

    init_times = ['00Z','06Z','12Z','18Z']

    for init_time in init_times:
        for f_hour in f_hours:

            load_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/0_GFS_downselect/%s/%s/'%(init_time,f_hour)
            files = sorted(os.listdir(load_dir))

            save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/CAPE_PR_model/%s/%s/'%(init_time,f_hour)
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)

            for f,load_file in enumerate(files):
                if f>=0:
                    ds = xr.open_dataset(load_dir+load_file,engine='netcdf4')
                    lat = ds['lat'].values
                    lon = ds['lon'].values
                    valid_times = ds['time'].values
                    cape = ds['cape'].values
                    precip_rate = ds['precip_rate'].values

                    ds2 = xr.Dataset(data_vars = dict(cape_pr = (["valid_times","lat","lon"],cape*precip_rate)),
                                    coords = dict(lon = (["lon"],lon),
                                                lat = (["lat"],lat),
                                                valid_times = (["valid_times"],valid_times)))

                    date_str = load_file[-9:-3]
                    save_file = 'BC_CAPE_PR_%s.nc'%(date_str)
                    print('saving: ',save_file)
                    ds2.to_netcdf(save_dir+save_file,engine='netcdf4')
                    del ds, ds2, load_file, date_str, cape, precip_rate, valid_times, lat, lon

if __name__=="__main__":
    main()