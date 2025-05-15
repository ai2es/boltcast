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

def calc_max_height(ds,var):

    """
    Calculate the maximum across the varying pressure levels. If there is only one level, 
    then the original dataset is returned. 
    """
    
    print("calculating the max for: " + var)
    if len(ds.dims)==4:#has multiple levels
        ds_max = ds.max(dim='levels')
        return ds_max
    else:#only has one level (i.e. CAPE, LI, precip_rate, reflectivity)
        return ds

def main():

    print('0b_stackedU_max_hgt.py main function')

    #declare the forecast days
    days = {'day_1':['f000','f003','f006','f009','f012','f015','f018','f021'],
            'day_2':['f024','f027','f030','f033','f036','f039','f042','f045'],
            'day_3':['f048','f051','f054','f057','f060','f063','f066','f069'],
            'day_4':['f072','f075','f078','f081','f084','f087','f090','f093']}

    #declare the variables and initialization times
    #get the initialization time.
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_times',type=int,default=1,help='Slurm array index for the init_times.')

    args = vars(parser.parse_args())
    init_times=['00Z','06Z','12Z','18Z']
    init_time = init_times[args['init_times']-1]

    var_list = ['cape','lifted_idx','reflectivity','precip_rate',
            'rain_q','ice_q','snow_q','graupel_q','w']

    """
    This portion of the code calculates the maximums across the variables, height wise. 
    It uses a self declared function.
    """
    for v,var in enumerate(var_list):
        for d,day in enumerate(days):
            f_hours = days[day]
            for h,f_hour in enumerate(f_hours):

                #declare the filenames and directories
                bc_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/3_vars_only/'+var+'/'
                print('max_height',h,f_hour,d,day,v,var,init_time)
                fname = '%s_%s_%s_%s.nc'%(day,var,f_hour,init_time)
                ds = xr.open_dataset(bc_dir+fname,engine='netcdf4')
                ds1 = calc_max_height(ds=ds,var=var)

                save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/4_vars_max_hgt/'+var+'/'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                ds1.to_netcdf(save_dir+fname,engine='netcdf4')
                del ds, ds1

if __name__=="__main__":
    main()