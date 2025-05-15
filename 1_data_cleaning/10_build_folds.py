import tensorflow as tf
import numpy as np
import os
import shutil
import pickle
import sys
import xarray as xr

def main():

    var = 'binary_ltg'
    ds_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/9_ds/'+var+'/'
    init_times = ['00Z','06Z','12Z','18Z']

    x_list = []
    y_list = []
    
    su_19 = slice('07/01/2019 00:00:00.000000000','08/31/2019 21:00:00.000000000')
    su_20 = slice('06/01/2020 00:00:00.000000000','08/31/2020 21:00:00.000000000')
    su_21 = slice('06/01/2021 00:00:00.000000000','08/31/2021 21:00:00.000000000')
    su_22 = slice('06/01/2022 00:00:00.000000000','08/31/2022 21:00:00.000000000')
    su_23 = slice('06/01/2023 00:00:00.000000000','08/31/2023 21:00:00.000000000')
    su_24 = slice('06/01/2024 00:00:00.000000000','06/30/2024 21:00:00.000000000')

    fa_19 = slice('09/01/2019 00:00:00.000000000','11/30/2019 21:00:00.000000000')
    fa_20 = slice('09/01/2020 00:00:00.000000000','11/30/2020 21:00:00.000000000')
    fa_21 = slice('09/01/2021 00:00:00.000000000','11/30/2021 21:00:00.000000000')
    fa_22 = slice('09/01/2022 00:00:00.000000000','11/30/2022 21:00:00.000000000')
    fa_23 = slice('09/01/2023 00:00:00.000000000','11/30/2023 21:00:00.000000000')

    sp_20 = slice('03/01/2020 00:00:00.000000000','05/31/2020 21:00:00.000000000')
    sp_21 = slice('03/01/2021 00:00:00.000000000','05/31/2021 21:00:00.000000000')
    sp_22 = slice('03/01/2022 00:00:00.000000000','05/31/2022 21:00:00.000000000')
    sp_23 = slice('03/01/2023 00:00:00.000000000','05/31/2023 21:00:00.000000000')
    sp_24 = slice('03/01/2024 00:00:00.000000000','05/31/2024 21:00:00.000000000')

    wi_20 = slice('12/01/2019 00:00:00.000000000','02/29/2020 21:00:00.000000000')
    wi_21 = slice('12/01/2020 00:00:00.000000000','02/28/2021 21:00:00.000000000')
    wi_22 = slice('12/01/2021 00:00:00.000000000','02/28/2022 21:00:00.000000000')
    wi_23 = slice('12/01/2022 00:00:00.000000000','02/28/2023 21:00:00.000000000')
    wi_24 = slice('12/01/2023 00:00:00.000000000','02/29/2024 21:00:00.000000000')

    seasons = {'summer':[su_19,su_20,su_21,su_22,su_23,su_24],
                'fall':[fa_19,fa_20,fa_21,fa_22,fa_23],
                'winter':[wi_20,wi_21,wi_22,wi_23,wi_24],
                'spring':[sp_20,sp_21,sp_22,sp_23,sp_24]}

    for season in seasons:
        print(season)
        season_slices = seasons[season]
        for init_time in init_times:

            fy = init_time+'_'+var+'_y.nc'
            fx = init_time+'_x.nc'

            x_ds = xr.open_dataset(ds_dir+fx,engine='netcdf4')
            y_ds = xr.open_dataset(ds_dir+fy,engine='netcdf4')

            for s in range(len(season_slices)):

                print(season_slices[s],)
                sn_sl = season_slices[s]

                sn_ds_x = x_ds.sel(valid_times=sn_sl)
                sn_ds_y = y_ds.sel(valid_times=sn_sl)
                
                sn_ds_x['y'] = sn_ds_y['y']
                tf_ds = tf.data.Dataset.from_tensor_slices((np.float32(sn_ds_x['x'].values),
                                                        np.float32(sn_ds_x['y'].values)))
                print(tf_ds)
                tf_save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_tf/'
                ds_save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/'
                fname = season+'_'+str(s)

                tf_ds.save(tf_save_dir+fname)
                sn_ds_x.to_netcdf(ds_save_dir+fname+'.nc',engine='netcdf4')
                del sn_ds_x, sn_ds_y, tf_ds
            del x_ds, y_ds

if __name__=="__main__":

    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print(n_visible_devices)
    tf.config.set_visible_devices([], 'GPU')
    print('NO VISIBLE DEVICES!!!!')

    main()