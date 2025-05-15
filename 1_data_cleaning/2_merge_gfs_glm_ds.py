import argparse
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import glob
import os

def match_glm():

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',type=str,default='2019',help='The year to run the code for.')
    args = vars(parser.parse_args())

    #declare the forecast hour based on the slurm array ID
    hours = []#int
    for i in range(33):
        hours.append(i*3)

    f_hours = []#string
    for hr in hours:
        f_hours.append('f'+f"{hr:03}")

    init_times = ['00Z','06Z','12Z','18Z']
    yr = args['year']

    badFiles = []

    # declare the months based on the years
    if yr=='2019':
        mos = ['07','08','09','10','11','12']
    elif yr=='2024':
        mos = ['01','02','03','04','05','06']
    else:
        mos=['01','02','03','04','05','06','07','08','09','10','11','12']

    for init_time in init_times:
        for f_hour in f_hours:
            for mo in mos:
                print("matching the glm and gfs data for: ")
                print(init_time, f_hour, yr, mo)

                #declare the forecast hour, directory of gfs_dir, and get the files
                gfs_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/0_GFS_downselect/%s/%s/'%(init_time,f_hour)
                gfs_file = 'BC_GFS_%s%s.nc'%(yr,mo)
                print(gfs_file)

                #declare the directory for the glm data 
                glm_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/1_GLM_downselect/CONUS/'

                #load the gfs data
                gfs_ds = xr.open_dataset(gfs_dir+gfs_file,engine='netcdf4')
                
                #get the gfs pressure levels and times
                levels = gfs_ds['levels'] #int
                gfs_valid_times = gfs_ds['time'].values #datetime64
                print(len(gfs_valid_times))

                binary_list = []
                fed_list = []
                num_bad_times=0

                #loop through the times to get the corresponding lightning data
                for t,time in enumerate(gfs_valid_times):
                    if t>=0:
                        #print(time)
                        #get the date information
                        time_ts = pd.Timestamp(time)
                        hour = time_ts.hour
                        day = time_ts.day
                        month = time_ts.month
                        year = time_ts.year

                        #declare the save directory and the file name
                        save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/2_merge_GFS_GLM/'+init_time+'/'+f_hour+'/'
                        if not os.path.isdir(save_dir):
                            os.makedirs(save_dir)
                        fname = 'BC_'+yr+'_'+mo+'.nc'

                        #load the pre-processed lightning data for the year
                        glm_ds = xr.open_dataset(glm_dir+f"{year:04}"+'.nc',engine='netcdf4')
                                        
                        try:
                            #get the lightning data for a specific time
                            ltg_fhour = glm_ds.sel(time=time)
                            glm_ds.close()
                        
                            #get the binary and fed data
                            binary_ltg = ltg_fhour['binary_ltg'].values #1x128x256
                            fed_ltg = ltg_fhour['FED'].values #1x128x256
                            
                            #make a 3-D tensor for the lightning data (lat,lon,time)
                            if t==0:
                                binary_stack = binary_ltg
                                fed_stack = fed_ltg
                            else:
                                binary_stack = np.dstack([binary_stack,binary_ltg])
                                fed_stack = np.dstack([fed_stack,fed_ltg])
                        except Exception as e:
                            print('bad valid time')
                            print(time)
                            print(e)
                            num_bad_times = num_bad_times+1
                            continue

                #this is to match the model times with the observations.
                #sometimes the model valid times exceed the lightning 
                #observation times, thise correct times
                if num_bad_times>0:
                    matched_times = gfs_valid_times[:-num_bad_times]
                    gfs_ds = gfs_ds.sel(time=matched_times)
                
                #get the axes in the correct order for storage in the xarray dataset
                binary_stack = np.swapaxes(np.swapaxes(binary_stack,0,2),1,2)
                fed_stack = np.swapaxes(np.swapaxes(fed_stack,0,2),1,2)

                print("here are the lightning shapes")
                print(binary_stack.shape)
                print(fed_stack.shape)
                print(fname)
                #add the lightning data to the dataset
                gfs_ds = gfs_ds.assign(binary_ltg=(["time","lat","lon"],binary_stack),fed=(["time","lat","lon"],fed_stack))
                gfs_ds.to_netcdf(save_dir+fname,engine='netcdf4',mode='w')

if __name__=='__main__':
    match_glm()