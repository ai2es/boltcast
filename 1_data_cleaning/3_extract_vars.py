import xarray as xr
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime, timedelta
import numpy as np
import argparse

"""
This file pulls the variables out then restores them into a netcdf file.  The netcdf file has the entire
period of interest in it, for each variable.  Here a variable is graupel_q, ice_q, w, reflectivity, etc. 

This allows for the stacked unet to be formulated easier.

The netcdf file has dimensions valid_time, lat, lon, height (if q/w) stored with the forecast time in the name
of the file.  
"""

def main():
    
    days = {'day_1':['f000','f003','f006','f009','f012','f015','f018','f021'],
            'day_2':['f024','f027','f030','f033','f036','f039','f042','f045'],
            'day_3':['f048','f051','f054','f057','f060','f063','f066','f069'],
            'day_4':['f072','f075','f078','f081','f084','f087','f090','f093']}

    #get the initialization time.
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_times',type=int,default=1,help='Slurm array index for the init_times.')
    args = vars(parser.parse_args())
    init_times=['00Z','06Z','12Z','18Z']
    init_time = init_times[args['init_times']-1]

    # var_list = ['cape','lifted_idx','reflectivity','precip_rate',
    #             'rain_q','ice_q','snow_q','graupel_q','w']

    var_list = ['fed','binary_ltg']
    
    for var in var_list:
        for d,day in enumerate(days):
            f_hours = days[day]
            for h,f_hour in enumerate(f_hours):
                var_data_list = []
                print(var,day,h,f_hour)
                bc_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/2_merge_GFS_GLM/%s/%s/'%(init_time,f_hour)
                files = sorted(os.listdir(bc_dir))
                for f,file in enumerate(files):
                    ds = xr.open_dataset(bc_dir+file,engine='netcdf4')
                    var_data = ds[var]
                    var_data = var_data.rename({'time':'valid_times'})
                    var_data_list.append(var_data)
                    del ds,var_data

                save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/3_vars_only/'+var+'/'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                fname = '%s_%s_%s_%s.nc'%(day,var,f_hour,init_time)
                ds2 = xr.concat(var_data_list,dim='valid_times')
                ds2.to_netcdf(save_dir+fname,engine='netcdf4')
                del ds2,var_data_list

if __name__=="__main__":
    main()