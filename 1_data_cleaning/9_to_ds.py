import numpy as np
import os
import xarray as xr

def main():
    print("building the tensorflow datasets")

    init_time = '18Z'

    x_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/6_vars_input/'

    var = 'fed'
    y_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/8_ltg_tf_ready/'+var+'/'

    cape_ds = xr.open_dataset(x_dir+'cape_'+init_time+'.nc',engine='netcdf4')
    lat = cape_ds['lat'].values
    lon = cape_ds['lon'].values
    days = cape_ds['days'].values

    vars = ['cape','lifted_idx','reflectivity','precip_rate','w','ice_q','snow_q','graupel_q','rain_q']

    li_ds = xr.open_dataset(x_dir+'lifted_idx_'+init_time+'.nc',engine='netcdf4')
    refl_ds = xr.open_dataset(x_dir+'reflectivity_'+init_time+'.nc',engine='netcdf4')
    pr_ds = xr.open_dataset(x_dir+'precip_rate_'+init_time+'.nc',engine='netcdf4')
    
    w_ds = xr.open_dataset(x_dir+'w_'+init_time+'.nc',engine='netcdf4')
    ice_ds = xr.open_dataset(x_dir+'ice_q_'+init_time+'.nc',engine='netcdf4')
    snow_ds = xr.open_dataset(x_dir+'snow_q_'+init_time+'.nc',engine='netcdf4')
    gr_ds = xr.open_dataset(x_dir+'graupel_q_'+init_time+'.nc',engine='netcdf4')
    rain_ds = xr.open_dataset(x_dir+'rain_q_'+init_time+'.nc',engine='netcdf4')

    ltg_ds = xr.open_dataset(y_dir+var+'_'+init_time+'.nc',engine='netcdf4')
    ltg_dates = ltg_ds['valid_times'].values
    
    input_list = []
    ltg_list = []
    time_list = []

    for ltg_vt in ltg_dates:

        ltg_np = ltg_ds.sel(valid_times=ltg_vt)['ltg_data'].values

        try:
            cape_np = cape_ds.sel(valid_times=ltg_vt)['var_data'].values
        except KeyError:
            print('bad cape')
            continue

        try:
            li_np = li_ds.sel(valid_times=ltg_vt)['var_data'].values
        except KeyError:
            print('bad li')
            continue
    
        try:
            pr_np = pr_ds.sel(valid_times=ltg_vt)['var_data'].values
        except KeyError:
            print('bad precip rate')
            continue

        try:
            refl_np = refl_ds.sel(valid_times=ltg_vt)['var_data'].values
        except KeyError:
            print('bad reflectivity')
            continue

        try:
            w_np = w_ds.sel(valid_times=ltg_vt)['var_data'].values
        except KeyError:
            print('bad w')
            continue
    
        try:
            ice_np = ice_ds.sel(valid_times=ltg_vt)['var_data'].values
        except KeyError:
            print('bad ice')
            continue

        try:
            snow_np = snow_ds.sel(valid_times=ltg_vt)['var_data'].values
        except KeyError:
            print('bad snow')
            continue

        try:
            graupel_np = gr_ds.sel(valid_times=ltg_vt)['var_data'].values
        except KeyError:
            print('bad graupel')
            continue
        
        try:
            rain_np = rain_ds.sel(valid_times=ltg_vt)['var_data'].values
        except KeyError:
            print('bad rain')
            continue

        #if all of the data are good for the lightning valid time
        uno_input_stack = np.stack([cape_np,li_np,refl_np,pr_np,w_np,ice_np,graupel_np,rain_np,snow_np],axis=3)

        time_list.append(ltg_vt)
        input_list.append(uno_input_stack)
        ltg_list.append(ltg_np)

    input_stack = np.stack(input_list,axis=0)
    output_stack = np.stack(ltg_list,axis=0)

    ds_features = xr.Dataset(data_vars = dict(x = (["valid_times","days","lat","lon","features"],input_stack)),
                            coords = dict(lon = (["lon"],lon),
                                            lat = (["lat"],lat),
                                            valid_times = (["valid_times"],time_list),
                                            days = (["days"],['day_1','day_2','day_3','day_4']),
                                            features = (["features"],vars)))
    
    ds_labels = xr.Dataset(data_vars = dict(y = (["valid_times","lat","lon","days"],output_stack)),
                                            coords=dict(lon = (["lon"],lon),
                                            lat = (["lat"],lat),
                                            valid_times = (["valid_times"],time_list),
                                            days = (["days"],['day_1','day_2','day_3','day_4'])))

    print('input_shape:',input_stack.shape)
    print('lightning_shape:',output_stack.shape)

    save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/9_ds/'+var+'/'
    fsave_x = init_time+'_x.nc'
    fsave_y = init_time+'_'+var+'_y.nc'
    
    ds_features.to_netcdf(save_dir+fsave_x,engine='netcdf4')
    ds_labels.to_netcdf(save_dir+fsave_y,engine='netcdf4')

if __name__=="__main__":
    main()