import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os

"""
This code generates netcdf files for each day of GLM data, within the 
CONUS BoltCast Domain. 
"""

#declare the satellite and the year (could use parser here)
sat = 'G17'
yr = '2021'

#build the julian days array in strings
j_days = []
for i in range(1,366):
    j_days.append(f"{i:03}")
if yr=='2020' or yr=='2024':
    j_days.append('366')

#go through the data for each day
for day in j_days:

    #declare the storage directory 
    glm_og_dir = '/ourdisk/hpc/ai2es/datasets/GLM/'+sat+'/'+yr+'/'+day+'/'
    
    #check if the data has been processed, if the storage file exists 
    #go to the next day
    save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/'+sat+'/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fsave = sat+'_'+yr+'_'+day+'_BC_df.nc'
    if os.path.isfile(save_dir+fsave):
        continue

    #get the glm source files from the directory
    #if the directory does not exist, continue on to the next day
    if os.path.isdir(glm_og_dir):
        glm_og_files = sorted(os.listdir(glm_og_dir))
    else:
        print('not a directory')
        print(glm_og_dir)
        continue

    #declare the coordinates you need from the glm data raw files
    coords = ['flash_lat','flash_lon','flash_id','flash_time_offset_of_first_event']#,'time_coverage_start','time_coverage_end']
    ltg_vars = ['flash_area','flash_energy','flash_quality_flag']
    df_cols = np.concatenate([coords,ltg_vars])

    #loop through the files
    first_file = True
    for f,file in enumerate(glm_og_files):
        if f>=0:#test control
            try:
                if f%100==0:#tracking 
                    print(file)

                #open the file, convert to dataframe, and extract the flash information 
                fname = glm_og_dir+file
                ds = xr.open_dataset(fname)
                ds = ds[ltg_vars]
                df = ds.to_dataframe()
                ds.close()
                df = df[df_cols] #subset the flash information with Lat/Lon
                df = df.loc[df['flash_quality_flag']==0]#get the good flashes

                #get the flashes within the latitude bounds of CONUS
                df = df.loc[df['flash_lat']>=21.25]
                df = df.loc[df['flash_lat']<=53]

                #get the west and east flashes
                if sat=='G16':#get the flashes east of 100W on GLM-Goes 16
                    df_BC = df.loc[df['flash_lon']>=-100]
                    df_BC = df_BC.loc[df_BC['flash_lon']<=-62.25]

                else: #get the flashes west of 100W on GLM-Goes 16
                    df_BC = df.loc[df['flash_lon']<-100]
                    df_BC = df_BC.loc[df_BC['flash_lon']>=-126]

                #convert the times to datetimes, create the index, and store the hour, minute, day, and year
                df_BC['date_time_UTC'] = pd.to_datetime(df_BC['flash_time_offset_of_first_event'])
                df_BC.index = df_BC['date_time_UTC']
                df_BC['minute'] = df_BC['date_time_UTC'].dt.minute
                df_BC['hour'] = df_BC['date_time_UTC'].dt.hour
                df_BC['day'] = df_BC['date_time_UTC'].dt.day
                df_BC['year'] = df_BC['date_time_UTC'].dt.year

                #append the dateframes together
                if first_file==True:
                    df_BC_day = df_BC
                    first_file=False
                else:
                    df_BC_day = pd.concat([df_BC_day,df_BC])

                #memory control
                del df, df_BC, ds
            except Exception as e:
                print(e)

    #sort the index and save to netcdf
    df_BC_day = df_BC_day.sort_index(axis=0)
    ds = df_BC_day.to_xarray()
    ds.to_netcdf(save_dir+fsave)

    #memory control
    del ds, df_BC_day


    





