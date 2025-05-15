import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime, timedelta
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import sys
import pickle

def main():

    print("6_make_BC_inputs.py")

    var_list = ['cape','lifted_idx','reflectivity','precip_rate',
            'rain_q','ice_q','snow_q','graupel_q','w']

    init_times = ['00Z','06Z','12Z','18Z']
    for init_time in init_times:
        for var in var_list:
            days = ['day_1','day_2','day_3','day_4']
            data_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/5_vars_max_day/'+var+'/'

            f1 = '%s_%s_%s_max_day.nc'%(days[0],var,init_time)
            f2 = '%s_%s_%s_max_day.nc'%(days[1],var,init_time)
            f3 = '%s_%s_%s_max_day.nc'%(days[2],var,init_time)
            f4 = '%s_%s_%s_max_day.nc'%(days[3],var,init_time)

            ds1 = xr.open_dataset(data_dir+f1,engine='netcdf4')
            lon = ds1['lon'].values
            lat = ds1['lat'].values

            ds2 = xr.open_dataset(data_dir+f2,engine='netcdf4')
            ds3 = xr.open_dataset(data_dir+f3,engine='netcdf4')
            ds4 = xr.open_dataset(data_dir+f4,engine='netcdf4')

            dt_24hr = np.timedelta64(24,'h')
            if init_time=='00Z':
                start1 = np.datetime64('2019-07-01 00:00:00.000000000')
                start2 = np.datetime64('2019-07-02 00:00:00.000000000')
                start3 = np.datetime64('2019-07-03 00:00:00.000000000')
                start4 = np.datetime64('2019-07-04 00:00:00.000000000')
            
            elif init_time=='06Z':
                start1 = np.datetime64('2019-07-01 06:00:00.000000000')
                start2 = np.datetime64('2019-07-02 06:00:00.000000000')
                start3 = np.datetime64('2019-07-03 06:00:00.000000000')
                start4 = np.datetime64('2019-07-04 06:00:00.000000000')
            
            elif init_time=='12Z':
                start1 = np.datetime64('2019-07-01 12:00:00.000000000')
                start2 = np.datetime64('2019-07-02 12:00:00.000000000')
                start3 = np.datetime64('2019-07-03 12:00:00.000000000')
                start4 = np.datetime64('2019-07-04 12:00:00.000000000')

            else:
                start1 = np.datetime64('2019-07-01 18:00:00.000000000')
                start2 = np.datetime64('2019-07-02 18:00:00.000000000')
                start3 = np.datetime64('2019-07-03 18:00:00.000000000')
                start4 = np.datetime64('2019-07-04 18:00:00.000000000')

            bad_date=False
            data_list = []
            time_list = []
            print(len(ds1['max_times']))
            for t in range(len(ds1['max_times'])):

                try:
                    data_np1 = ds1.sel(max_times=start1)['max_var'].values
                except KeyError:
                    print('bad dates: ',start1)
                    bad_date=True

                try:
                    data_np2 = ds2.sel(max_times=start2)['max_var'].values
                except KeyError:
                    print('bad dates: ',start2)
                    bad_date=True

                try:
                    data_np3 = ds3.sel(max_times=start3)['max_var'].values
                except KeyError:
                    print('bad dates: ',start3)
                    bad_date=True
                
                try:
                    data_np4 = ds4.sel(max_times=start4)['max_var'].values
                except KeyError:
                    # print('bad dates: ',start4)
                    bad_date=True

                if bad_date==True:
                    pass

                else: 
                    data_stack = np.dstack([data_np1,data_np2,data_np3,data_np4])#128,256,4
                    data_stack = np.swapaxes(np.swapaxes(data_stack,2,0),1,2)#4,128,256  
                    # data_stack = np.expand_dims(data_stack,axis=3)#4,128,256,1
                    # data_stack = np.expand_dims(data_stack,axis=0)#1,4,128,256,1

                    time_list.append(start1)
                    data_list.append(data_stack)
                
                start1 = start1+dt_24hr
                start2 = start2+dt_24hr
                start3 = start3+dt_24hr
                start4 = start4+dt_24hr

                bad_date=False

            data_batch = np.stack(data_list,axis=0)
            print(var, init_time, data_batch.shape)

            ds_var99 = xr.Dataset(data_vars = dict(var_data = (["valid_times","days","lat","lon"],data_batch)),
                            coords = dict(lon = (["lon"],lon),
                                            lat = (["lat"],lat),
                                            valid_times = (["valid_times"],time_list),
                                            days = (["days"],['day_1','day_2','day_3','day_4'])))
            
            fsave = '%s_%s.nc'%(var,init_time)
            save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/6_vars_input/'
            print('saving '+fsave)
            ds_var99.to_netcdf(save_dir+fsave,engine='netcdf4')
            # print(f"{sys.getsizeof(data_batch)*1e-9:.2f}"+'GB')
            # var_dict = {'time':time_list,
            #             'data':data_batch}
            # pickle.dump(var_dict,open('/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/6_vars_input/'+var+'_'+init_time+'.pkl','wb'))


if __name__=='__main__':
    main()