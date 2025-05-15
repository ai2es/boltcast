import numpy as np
import xarray as xr
import os 
import shutil 
import glob
import numpy as np
import matplotlib.pyplot as plt
import pygrib
import pickle
import argparse
import wandb

"""
This file has code to downselect the GFS data to a certain domain, then save that off.  
"""

"""
This function gets the graupel mixing ratios in kg/kg from the gfs grib files. 
"""

def get_variable(label, levels, grbindx, CONUS):
    for i,level in enumerate(levels):
        if i==0:
            data_3d = grbindx.select(name=label,typeOfLevel='isobaricInhPa',level=level)[0].values[CONUS[0]:CONUS[1],CONUS[2]:CONUS[3]]
        else:
            data_3d = np.dstack([data_3d, 
                                    grbindx.select(name=label,typeOfLevel='isobaricInhPa',level=level)[0].values[CONUS[0]:CONUS[1],CONUS[2]:CONUS[3]]])
    return data_3d

def gfs_open():

    #create a parser for the initialization time and the forecast hour for easy HPC use
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id',type=int,default=1,help='Slurm array index for the job.')
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

    #declare the months based on the years
    if yr=='2019':
        mos = ['07','08','09','10','11','12']
    elif yr=='2024':
        mos = ['01','02','03','04','05','06']
    else:
        mos=['01','02','03','04','05','06','07','08','09','10','11','12']

    job_dict = {}
    job_id = 1
    for init_time in init_times:
        for f_hour in f_hours:
            for mo in mos:
                job_dict[job_id] = [init_time,f_hour,yr,mo]
                job_id = job_id+1
                print(job_id)
                
    idx = args['job_id']
    init_time = job_dict[idx][0]
    f_hour = job_dict[idx][1]
    yr = job_dict[idx][2]
    mo = job_dict[idx][3]

    print('downselecting the gfs data and variables for: ')
    print(init_time, f_hour, yr, mo)

    run = wandb.init(project='BoltCast_GFS_downselect',
                    entity='bmac7167', 
                    name=init_time+'_'+f_hour, 
                    dir='/scratch/bmac87/wandb', 
                    job_type='GFS_downselect')

    #declare the output directory
    save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/0_GFS_downselect/'+init_time+'/'+f_hour+'/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    
    #declare where the GFS grib files are stored
    gfs_dir = '/ourdisk/hpc/ai2es/datasets/GFS/'+init_time+'/'+f_hour+'/'
        
    #these indices are used to downselect the CONUS domain
    north_idx=148
    south_idx=276
    west_idx=936
    east_idx=1192

    CONUS = [north_idx,south_idx,west_idx,east_idx]

    #these are the levels in hPa where we want the environmental information 
    levels = [200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,925,950,975,1000]

    #save of the 1-D lat and lon grids
    gfs_grid = pickle.load(open('../2_model_training/static_inputs.pkl','rb'))
    boltCast_lon_1d = gfs_grid['lon'][0,:]
    boltCast_lat_1d = gfs_grid['lat'][:,0]
    
    #get the file list for each year and month. 
    if os.path.isdir(gfs_dir):
        file_list = sorted(glob.glob(gfs_dir+'/gfs.0p25.'+yr+mo+'*.'+f_hour+'.grib2'))
    else: 
        print('not directory')
        print(gfs_dir)

    #declare lists for the time, data, bad files
    timeList = []
    dataList = []

    print(yr,mo,f_hour)

    for file in file_list:
        print(file)
        try:
            grbs = pygrib.open(file)
            valid_time = grbs[1].validDate
            
            #get the variables across all of the levels
            grbindx = pygrib.index(file,'name','typeOfLevel','level')
            graupel_3d = get_variable('Graupel (snow pellets)',levels,grbindx,CONUS)
            ice_3d = get_variable('Ice water mixing ratio',levels,grbindx,CONUS)
            rain_3d = get_variable('Rain mixing ratio',levels,grbindx,CONUS)
            snow_3d = get_variable('Snow mixing ratio',levels,grbindx,CONUS)
            w_3d = get_variable('Geometric vertical velocity',levels,grbindx,CONUS)

            #get the accumulated atmospheric variables
            reflectivity = grbs.select(name='Maximum/Composite radar reflectivity',typeOfLevel='atmosphere',level=0)[0].values[north_idx:south_idx,west_idx:east_idx]
            cape = grbs.select(name='Convective available potential energy',typeOfLevel='surface',level=0)[0].values[north_idx:south_idx,west_idx:east_idx]
            lifted_idx = grbs.select(name='Surface lifted index',typeOfLevel='surface',level=0)[0].values[north_idx:south_idx,west_idx:east_idx]
            precip_rate = grbs.select(name='Precipitation rate',typeOfLevel='surface',level=0)[0].values[north_idx:south_idx,west_idx:east_idx]
            
            ds = xr.Dataset(
                data_vars = dict(
                    
                    graupel_q = (["lat","lon","levels"],graupel_3d),
                    ice_q = (["lat","lon","levels"],ice_3d),
                    rain_q = (["lat","lon","levels"],rain_3d),
                    snow_q = (["lat","lon","levels"],snow_3d),
                    
                    w = (["lat","lon","levels"],w_3d),

                    reflectivity = (["lat","lon"],reflectivity),
                    cape = (["lat","lon"],cape),
                    lifted_idx = (["lat","lon"],lifted_idx),
                    precip_rate = (["lat","lon"],precip_rate)
                ),

                coords = dict(
                    lon = (["lon"],boltCast_lon_1d),
                    lat = (["lat"],boltCast_lat_1d),
                    levels = (["levels"],levels)
                ),

                attrs = dict(
                    description="Inputs into BoltCast.",
                    mixing_ratio_units="kg/kg",
                    refl_units="dB",
                    cape_units="J/kg",
                    lifted_index_units="K",
                    precip_rate_units="kg/(m^2 s)",
                    w_units="m/s",
                    time="datetime in UTC/Zulu time zone",
                    lat_lon="degrees",
                    levels="Index 0 is 200mb, while index 19 is 1000mb.  In other words the first index is the highest level in the atmosphere."
                )
            )
            timeList.append(valid_time)
            dataList.append(ds)
        #end_try

        except Exception as e: 
            print("Exception when down selecting GFS data")
            print(valid_time)
            print(file)
            badFiles.append(file)
            print(e)
            print()
            continue
        #end_except
    #end_for_files

    #build the final dataset from the dataList variable
    try:
        print("saving the data to ourdisk")
        ds1 = xr.concat(dataList, data_vars='all', dim='time')
        ds1 = ds1.assign_coords(time=timeList)
        ds1 = ds1.sortby('time')

        #create the save directory and filename
        if os.path.isdir(save_dir) == False:
            print("creating the save directory")
            os.makedirs(save_dir)
        fname = 'BC_GFS_%s%s.nc'%(yr,mo)
        print(save_dir+fname)
        ds1.to_netcdf(save_dir+fname,engine='netcdf4',mode='w')
        del ds1, dataList, timeList, ds

    except Exception as e:
        print()
        print("Exception when concatenating the data. Most likely, the files don't exist for:")
        print(e)
        print(mo, yr)

    print()
    print("Bad Files: ")
    bad_grib_txt = open('./bad_files/bad_gribs_'+yr+mo+f_hour+'.txt','w')
    for badFile in badFiles:
        bad_grib_txt.write(badFile)
        bad_grib_txt.write('\n')
    bad_grib_txt.close()
    print(badFiles)
    print()
    pickle.dump(badFiles,open('./bad_files/bad_gribs_'+yr+mo+f_hour+'.pkl','wb'))

    run.finish()
#end gfs_open()

if __name__=="__main__":
    gfs_open()
