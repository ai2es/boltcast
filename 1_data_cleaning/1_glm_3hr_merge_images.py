import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os 
import shutil 
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygrib
import pickle
import datetime
import argparse 
from util import boxbin, get_day_mo

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=  18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

def plot_lightning(annual_ltg):

    valid_times = annual_ltg['time'].values
    lon = annual_ltg['lon'].values
    lat = annual_ltg['lat'].values

    for i,time in enumerate(valid_times):

        if i>=0:
            ltg_data = annual_ltg.sel(time=time)
            fed = ltg_data['FED'].values
            binary_ltg = ltg_data['binary_ltg'].values

            fig,axs = plt.subplots(nrows=1,
                                    ncols=2, 
                                    subplot_kw={'projection': ccrs.PlateCarree()})

            axs[0].add_feature(cfeature.COASTLINE,edgecolor="black")
            axs[0].add_feature(cfeature.LAND)
            axs[0].add_feature(cfeature.OCEAN)
            axs[0].add_feature(cfeature.STATES,edgecolor="black")
            axs[0].set_title("Flash Extent Density")
            cb0 = axs[0].pcolormesh(lon,lat,fed)

            axs[1].add_feature(cfeature.COASTLINE,edgecolor="black")
            axs[1].add_feature(cfeature.LAND)
            axs[1].add_feature(cfeature.OCEAN)
            axs[1].add_feature(cfeature.STATES,edgecolor="black")
            axs[1].set_title("Binary Classification")
            cb1 = axs[1].pcolormesh(lon,lat,binary_ltg,cmap='binary')

            plt.suptitle(time)

            valid_time = pd.Timestamp(time)
            hour = valid_time.hour
            month = valid_time.month
            year = valid_time.year
            day = valid_time.day

            image_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/test_images/'
            fsave = 'GLM_%s%s%s%sZ.png'%(year,f"{month:02}",f"{day:02}",f"{hour:02}")
            plt.savefig(image_dir+fsave)
            plt.close()

def main():

    year = '2019'
    nc_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/CONUS/'
    annual_ltg = xr.open_dataset(nc_dir+year+'.nc',engine='netcdf4')
    plot_lightning(annual_ltg)


if __name__=='__main__':
    main()