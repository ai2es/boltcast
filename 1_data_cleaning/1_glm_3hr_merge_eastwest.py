import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygrib
import pickle
import datetime
import argparse
import xarray as xr 
from util import boxbin, get_day_mo

def main():
    
    year = '2024'
    if year=='2023' or year=='2024':
        w_sat = 'G18'
    else:
        w_sat = 'G17'

    e_sat = 'G16'

    #load the gfs grid
    static_inputs = pickle.load(open('./Archive/static_inputs.pkl','rb'))
    lat = static_inputs['lat']#2D
    lon = static_inputs['lon']#2D

    #add the next grid point so the binning algorithm 
    #size is 256x128
    bin_lat = np.concatenate([lat[:,0],[53.25]])
    bin_lon = np.concatenate([lon[0,:],[298]])

    #store the sorted grid
    xedge = np.sort(bin_lon)
    yedge = np.sort(bin_lat)

    xmid = [] #Blank array
    ymid = [] #Blank array

    #calcuate the midpoints for the sorting algorithm
    i=0
    while(i < len(xedge)-1):
        xmid.append((xedge[i]+xedge[i+1])/2) #Calculate and append midpoints
        i+=1 
    i=0
    while(i < len(yedge)-1):
        ymid.append((yedge[i]+yedge[i+1])/2) #Calculate and append midpoints
        i+=1

    #build the array of julian days. add 366 for leap year
    j_days = []
    for i in range(1,183):
        j_days.append(f"{i:03}")
    if year=='2020':
        j_days.append('366')

    #declare the data directories
    e_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/%s/'%(e_sat)
    w_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/%s/'%(w_sat)
    
    data_list = []

    for day in j_days:

        print(day)

        #declare the file names
        e_file = '%s_%s_%s_BC_ds.nc'%(e_sat,year,day)
        w_file = '%s_%s_%s_BC_ds.nc'%(w_sat,year,day)

        #load the data
        e_ds = xr.open_dataset(e_dir+e_file,engine='netcdf4')
        w_ds = xr.open_dataset(w_dir+w_file,engine='netcdf4')
        
        #convert nans to zeros for the fed
        e_fed = np.nan_to_num(e_ds['FED'].values)
        w_fed = np.nan_to_num(w_ds['FED'].values)

        #extract the binary lightning data
        e_binary = e_ds['binary_ltg'].values
        w_binary = w_ds['binary_ltg'].values

        #add the two (numpy arrays)
        conus_fed = e_fed+w_fed
        conus_binary = e_binary+w_binary

        #store the counts and binary classification into an xarray dataset
        ds = xr.Dataset(
                data_vars=dict(binary_ltg = (["time","lat","lon"],conus_binary),
                                FED = (["time","lat","lon"],conus_fed)),
                coords=dict(lon=(["lon"],xmid),lat=(["lat"],ymid),time=(["time"],e_ds['time'].values)),
                attrs=dict(description="Binary classification and FED of GLM on GFS grid. 256x128.",
                            FED="number of GLM flashes in GFS grid box, currently just a number",
                            binary_ltg="0 or 1 depicting if one lightning flash occurred in the grid box")
            )

        data_list.append(ds)

        #memory control
        del ds, e_ds, w_ds, conus_fed, conus_binary

    #concatenate all of the datasets along the time dimension
    annual_ds = xr.concat(data_list, dim="time")
    print(annual_ds)

    save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/CONUS/'
    annual_ds.to_netcdf(save_dir+year+'.nc',engine='netcdf4')

    

if __name__=='__main__':
    main()