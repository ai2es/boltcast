import os
import xarray as xr
import numpy as np
import pickle

def main():
    
    print("building lightning outputs")
    
    init_times = ['00Z','06Z','12Z','18Z']
    init_time = init_times[3]
    var = 'fed'

    ltg_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/7_outputs/'+var+'/'
    files = sorted(os.listdir(ltg_dir))
    
    f1 = 'day_1_'+var+'_'+init_time+'.nc'
    f2 = 'day_2_'+var+'_'+init_time+'.nc'
    f3 = 'day_3_'+var+'_'+init_time+'.nc'
    f4 = 'day_4_'+var+'_'+init_time+'.nc'

    ds1 = xr.open_dataset(ltg_dir+f1,engine='netcdf4')
    lat = ds1['lat'].values
    lon = ds1['lon'].values

    ds2 = xr.open_dataset(ltg_dir+f2,engine='netcdf4')
    ds3 = xr.open_dataset(ltg_dir+f3,engine='netcdf4')
    ds4 = xr.open_dataset(ltg_dir+f4,engine='netcdf4')

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

    data_list = []
    time_list = []

    for i in range(1830):
        # if i%100==0:
        #     print(start1)
        try:
            ltg1_np = ds1.sel(ltg_times=start1)['ltg_data'].values
        except KeyError:
            start1 = start1+dt_24hr
            start2 = start2+dt_24hr
            start3 = start3+dt_24hr
            start4 = start4+dt_24hr
            continue
        
        try:
            ltg2_np = ds2.sel(ltg_times=start2)['ltg_data'].values
        except KeyError:
            start1 = start1+dt_24hr
            start2 = start2+dt_24hr
            start3 = start3+dt_24hr
            start4 = start4+dt_24hr
            continue

        try:
            ltg3_np = ds3.sel(ltg_times=start3)['ltg_data'].values
        except KeyError:
            start1 = start1+dt_24hr
            start2 = start2+dt_24hr
            start3 = start3+dt_24hr
            start4 = start4+dt_24hr
            continue

        try:
            ltg4_np = ds4.sel(ltg_times=start4)['ltg_data'].values     
        except KeyError:
            start1 = start1+dt_24hr
            start2 = start2+dt_24hr
            start3 = start3+dt_24hr
            start4 = start4+dt_24hr
            continue

        ltg_4days = np.dstack([ltg1_np,ltg2_np,ltg3_np,ltg4_np])

        data_list.append(ltg_4days)
        time_list.append(start1)
        
        start1 = start1+dt_24hr
        start2 = start2+dt_24hr
        start3 = start3+dt_24hr
        start4 = start4+dt_24hr

    data_batch = np.stack(data_list,axis=0)
    save_dir =  '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/8_ltg_tf_ready/'+var+'/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    fsave = '%s_%s.nc'%(var,init_time)

    ds_ltg = xr.Dataset(data_vars = dict(ltg_data = (["valid_times","lat","lon","days"],data_batch)),
                            coords = dict(lon = (["lon"],lon),
                                            lat = (["lat"],lat),
                                            valid_times = (["valid_times"],time_list),
                                            days = (["days"],['day_1','day_2','day_3','day_4'])))
    ds_ltg.to_netcdf(save_dir+fsave,engine='netcdf4')

if __name__=="__main__":
    main()