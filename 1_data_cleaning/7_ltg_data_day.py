import xarray as xr
import numpy as np
import pickle
import os
import shutil
import pandas
import matplotlib.pyplot as plt

def calc_lightning_per_day(f_hour_dict,ltg_var,init_time,day):

    print("building the lightning data for: ",ltg_var,init_time)

    #declare the important time changes
    dt_3hr = np.timedelta64(3,'h')
    dt_24hr = np.timedelta64(24,'h')

    days = {'day_1':['f000','f003','f006','f009','f012','f015','f018','f021'],
            'day_2':['f024','f027','f030','f033','f036','f039','f042','f045'],
            'day_3':['f048','f051','f054','f057','f060','f063','f066','f069'],
            'day_4':['f072','f075','f078','f081','f084','f087','f090','f093']}

    #load the datasets for the different forecast hours
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

    #declare the start dates for each time
    if init_time=='00Z':
        if day=='day_1':
            start = np.datetime64('2019-07-01 00:00:00.000000000')
        elif day=='day_2':
            start = np.datetime64('2019-07-02 00:00:00.000000000')
        elif day=='day_3':
            start = np.datetime64('2019-07-03 00:00:00.000000000')
        else:
            start = np.datetime64('2019-07-04 00:00:00.000000000')

    elif init_time=='06Z':
        if day=='day_1':
            start = np.datetime64('2019-07-01 06:00:00.000000000')
        elif day=='day_2':
            start = np.datetime64('2019-07-02 06:00:00.000000000')
        elif day=='day_3':
            start = np.datetime64('2019-07-03 06:00:00.000000000')
        else:
            start = np.datetime64('2019-07-04 06:00:00.000000000')

    elif init_time=='12Z':
        if day=='day_1':
            start = np.datetime64('2019-07-01 12:00:00.000000000')
        elif day=='day_2':
            start = np.datetime64('2019-07-02 12:00:00.000000000')
        elif day=='day_3':
            start = np.datetime64('2019-07-03 12:00:00.000000000')
        else:
            start = np.datetime64('2019-07-04 12:00:00.000000000')

    else:
        if day=='day_1':
            start = np.datetime64('2019-07-01 18:00:00.000000000')
        elif day=='day_2':
            start = np.datetime64('2019-07-02 18:00:00.000000000')
        elif day=='day_3':
            start = np.datetime64('2019-07-03 18:00:00.000000000')
        else:
            start = np.datetime64('2019-07-04 18:00:00.000000000')

    time_list = []
    data_list = []

    for i in range(1830):
        if i%100==0:
            pass
            # print(start)
        try: 
            data0 = ds0.sel(valid_times=start)[ltg_var].values
        except KeyError:
            print('bad date in ds0: for',start)
            start = start+dt_24hr
            continue

        try: 
            data1 = ds1.sel(valid_times=start+dt_3hr)[ltg_var].values
        except KeyError:
            print('bad date in ds1: for',start+dt_3hr)
            start = start+dt_24hr
            continue

        try: 
            data2 = ds2.sel(valid_times=start+(dt_3hr*2))[ltg_var].values
        except KeyError:
            print('bad date in ds2: for',start+(dt_3hr*2))
            start = start+dt_24hr
            continue

        try: 
            data3 = ds3.sel(valid_times=start+(dt_3hr*3))[ltg_var].values
        except KeyError:
            print('bad date in ds3: for',start+(dt_3hr*3))
            start = start+dt_24hr
            continue

        try: 
            data4 = ds4.sel(valid_times=start+(dt_3hr*4))[ltg_var].values
        except KeyError:
            print('bad date in ds4: for',start+(dt_3hr*4))
            start = start+dt_24hr
            continue

        try: 
            data5 = ds5.sel(valid_times=start+(dt_3hr*5))[ltg_var].values
        except KeyError:
            print('bad date in ds5: for',start+(dt_3hr*5))
            start = start+dt_24hr
            continue

        try: 
            data6 = ds6.sel(valid_times=start+(dt_3hr*6))[ltg_var].values
        except KeyError:
            print('bad date in ds6 for',start+(dt_3hr*6))
            start = start+dt_24hr
            continue

        try: 
            data7 = ds7.sel(valid_times=start+(dt_3hr*7))[ltg_var].values
        except KeyError:
            print('bad date in ds7 for',start+(dt_3hr*7))
            start = start+dt_24hr
            continue
        
        
        day_ltg = np.stack([data0, data1, data2, data3, data4, data5, data6, data7],axis=2)

        if ltg_var == 'fed':
            day_ltg = np.sum(day_ltg,axis=2)
        else:
            day_ltg = np.sum(day_ltg,axis=2)
            day_ltg[day_ltg>=1] = 1

        ds_ltg = xr.Dataset(data_vars = dict(ltg_data = (["lat","lon"],day_ltg)),
                            coords = dict(lon = (["lon"],lon),lat = (["lat"],lat)))

        data_list.append(ds_ltg)
        time_list.append(start)

        start = start+dt_24hr

    ds = xr.concat(data_list, data_vars='all', dim='ltg_times')
    ds = ds.assign_coords(ltg_times=time_list)
    ds = ds.sortby('ltg_times')
    return ds

def main():

    days = {'day_1':['f000','f003','f006','f009','f012','f015','f018','f021'],
            'day_2':['f024','f027','f030','f033','f036','f039','f042','f045'],
            'day_3':['f048','f051','f054','f057','f060','f063','f066','f069'],
            'day_4':['f072','f075','f078','f081','f084','f087','f090','f093']}

    init_times = ['00Z','06Z','12Z','18Z']

    ltg_vars = ['fed']

    for init_time in init_times:
        for ltg,ltg_var in enumerate(ltg_vars):
            ltg_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/3_vars_only/'+ltg_var+'/'
            for d,day in enumerate(days):
                f_hours = days[day]
                f_hour_dict = {}
                for h,f_hour in enumerate(f_hours):
                    ltg_data_list = []
                    fname = '%s_%s_%s_%s.nc'%(day,ltg_var,f_hour,init_time)
                    ds = xr.open_dataset(ltg_dir+fname,engine='netcdf4')
                    f_hour_dict[f_hour] = ds
                    del ds

                fsave = '%s_%s_%s.nc'%(day,ltg_var,init_time)
                print(fsave)
                ds_save = calc_lightning_per_day(f_hour_dict=f_hour_dict,ltg_var=ltg_var,init_time=init_time,day=day)
                print('start_time')
                print(ds_save['ltg_times'].values[0])
                print('end_time')
                print(ds_save['ltg_times'].values[-1])
                print('number_of_times')
                print(len(ds_save['ltg_times'].values))

                save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/7_outputs/'+ltg_var+'/'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                ds_save.to_netcdf(save_dir+fsave,engine='netcdf4')
                del ds_save, f_hour_dict

if __name__=='__main__':
    main()