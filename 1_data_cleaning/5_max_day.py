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
import pandas as pd

"""
This file calculates the maxes with height given an xarray dataset, ds, and the 
variable name. 

This functions takes in a dictionary containing xarray datasets for each forecast hour 
(f_hour_dict), for the given variable var. 

This function calculates the maximum over the forecasted day. 
"""
def calc_max_day(f_hour_dict,var,day,init_time):

    #create the dataset to store the maxes in (should be the same size as f000, f024, f048, f072, f096)
    days = {'day_1':['f000','f003','f006','f009','f012','f015','f018','f021'],
            'day_2':['f024','f027','f030','f033','f036','f039','f042','f045'],
            'day_3':['f048','f051','f054','f057','f060','f063','f066','f069'],
            'day_4':['f072','f075','f078','f081','f084','f087','f090','f093']}

    f_hours = days[day]
    ds0 = f_hour_dict[f_hours[0]]
    lat = ds0['lat'].values
    lon = ds0['lon'].values

    ds1 = f_hour_dict[f_hours[1]]
    ds2 = f_hour_dict[f_hours[2]]
    ds3 = f_hour_dict[f_hours[3]]
    ds4 = f_hour_dict[f_hours[4]]
    ds5 = f_hour_dict[f_hours[5]]
    ds6 = f_hour_dict[f_hours[6]]
    ds7 = f_hour_dict[f_hours[7]]

    if init_time=='00Z':
        start_time = np.datetime64('2019-07-01 00:00:00.000000')
    if init_time=='06Z':
        start_time = np.datetime64('2019-07-01 06:00:00.000000')
    if init_time=='12Z':
        start_time = np.datetime64('2019-07-01 12:00:00.000000')
    if init_time=='18Z':
        start_time = np.datetime64('2019-07-01 18:00:00.000000')

    dt_3hr = np.timedelta64(3,'h')
    dt_24hr = np.timedelta64(24,'h')

    #make a dataframe for the expected datetimes
    if day=='day_1':
        num_days = 1827

    if day=='day_2':
        start_time=start_time+dt_24hr
        num_days = 1826

    if day=='day_3':
        start_time=start_time+dt_24hr+dt_24hr
        num_days = 1825

    if day=='day_4':
        start_time=start_time+dt_24hr+dt_24hr+dt_24hr
        num_days = 1824
    
    #build the ideal valid times
    ideal_dates = []
    for r in range(num_days):#ends on June 30 2024
        row = []
        for c in range(8):
            row.append(start_time)
            start_time=start_time+dt_3hr
        ideal_dates.append(row)

    data_list = []
    time_list = []
    except_ct = 0

    for r in range(num_days):   
        dates = ideal_dates[r]
        
        try:
            data0 = ds0.sel(valid_times=dates[0])[var].values
        except KeyError:
            print('bad date:',dates[0])
            continue
        
        try:
            data1 = ds1.sel(valid_times=dates[1])[var].values
        except KeyError:
            print('bad date:',dates[1])
            continue

        try:
            data2 = ds2.sel(valid_times=dates[2])[var].values
        except KeyError:
            print('bad date:',dates[2])
            continue

        try:
            data3 = ds3.sel(valid_times=dates[3])[var].values
        except KeyError:
            print('bad date:',dates[3])
            continue

        try:
            data4 = ds4.sel(valid_times=dates[4])[var].values
        except KeyError:
            print('bad date:',dates[4])
            continue
        
        try:
            data5 = ds5.sel(valid_times=dates[5])[var].values
        except KeyError:
            print('bad date:',dates[5])
            continue
    
        try:
            data6 = ds6.sel(valid_times=dates[6])[var].values
        except KeyError:
            print('bad date:',dates[6])
            continue

        try:
            data7 = ds7.sel(valid_times=dates[7])[var].values
        except KeyError:
            print('bad date:',dates[7])
            continue

        print("hooray, the forecast day is all there, starting with:")
        print(dates[0])

        data_stack = np.dstack([data0,data1,data2,data3,data4,data5,data6,data7])
        data_max = np.max(data_stack,axis=2)

        ds_max = xr.Dataset(data_vars = dict(max_var = (["lat","lon"],data_max)),
                            coords = dict(lon = (["lon"],lon),lat = (["lat"],lat)))

        data_list.append(ds_max)
        time_list.append(dates[0])

    ds_maxes = xr.concat(data_list, data_vars='all', dim='max_times')
    ds_maxes = ds_maxes.assign_coords(max_times=time_list)
    ds_maxes = ds_maxes.sortby('max_times')

    return ds_maxes

def load_f_hour_dict(f_hours,var,init_time,day):
    f_hour_dict = {}#dictionary to hold the datasets of the forecast hours of each forecast day

    #load the data across the 24-hour forecast period 
    for h,f_hour in enumerate(f_hours):
        print(h,f_hour)
        max_hgt_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/4_vars_max_hgt/'+var+'/'
        fname = '%s_%s_%s_%s.nc'%(day,var,f_hour,init_time)
        
        ds = xr.open_dataset(max_hgt_dir+fname,engine='netcdf4')
        f_hour_dict[f_hour] = ds
        del ds

    return f_hour_dict          


def make_test_images(ds,var):
    print("making test images for: ",var)
    print(ds)
    
    lat = ds['lat']
    lon = ds['lon']
    if var=='fed' or var=='binary_ltg':
        valid_times = ds['ltg_times']
    else:
        valid_times = ds['max_times']

    for t,time in enumerate(valid_times):
        if t%300==0:
            print(time)
            if var=='fed' or var=='binary_ltg':
                data = ds.sel(ltg_times=time)
                data_np = data['ltg_data'].values
            else:
                data = ds.sel(max_times=time)
                data_np = data['max_var'].values

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
            cb = ax.pcolormesh(lon,lat,data_np,transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
            ax.add_feature(cfeature.STATES,edgecolor="gray")
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='white', alpha=0.5, linestyle='--')
            plt.colorbar(cb)
            
            ts = pd.Timestamp(time.values)
            hour = ts.hour
            day = ts.day
            month = ts.month
            year = ts.year
            time_str = '%s_%s_%s_%sZ'%(month,day,year,f"{hour:02}")
            image_dir = './max_images/'+var+'/'
            if os.path.isdir(image_dir)==False:
                os.makedirs(image_dir)
            plt.suptitle('24hr Max '+var+': '+time_str)
            plt.savefig(image_dir+var+'_'+time_str+'.png')
            plt.close()
            
    del data, ds 

def main():

    print('5_max_day.py main function')

    #declare the forecast days
    days = {'day_1':['f000','f003','f006','f009','f012','f015','f018','f021'],
            'day_2':['f024','f027','f030','f033','f036','f039','f042','f045'],
            'day_3':['f048','f051','f054','f057','f060','f063','f066','f069'],
            'day_4':['f072','f075','f078','f081','f084','f087','f090','f093']}

    #declare the variables and initialization times get the initialization time.
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_times',type=int,default=1,help='Slurm array index for the init_times.')
    args = vars(parser.parse_args())
    init_times=['00Z','06Z','12Z','18Z']
    init_time = init_times[args['init_times']-1]

    var_list = ['cape','lifted_idx','reflectivity','precip_rate',
                'rain_q','ice_q','snow_q','graupel_q','w']

    """
    This portion of the code calculates the maximums across the variables, day wise. 
    It uses a self declared function.
    """

    for v,var in enumerate(var_list):
        for d,day in enumerate(days):
            if d>=0:
                print(var,day,init_time)

                f_hours = days[day]
                f_hour_dict = load_f_hour_dict(f_hours=f_hours,var=var,init_time=init_time,day=day)
                save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/5_vars_max_day/'+var+'/'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                fname_save = '%s_%s_%s_max_day.nc'%(day,var,init_time)

                print(fname_save)
                ds_max = calc_max_day(f_hour_dict=f_hour_dict,var=var, day=day,init_time=init_time)
                # make_test_images(ds_max,var)
                ds_max.to_netcdf(save_dir+fname_save,engine='netcdf4')

                del f_hour_dict, ds_max

if __name__=="__main__":
    main()