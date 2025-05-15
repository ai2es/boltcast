import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os 
import shutil 
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import datetime as dt

#################################################################
# Default plotting parameters
FIGURESIZE=(8,8)
FONTSIZE=8
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

year = '2020'
month = '06'
f_hour = 'f096'
init_time = '06Z'

gfs_data = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/0_GFS_downselect/%s/%s/BC_GFS_%s%s.nc'%(init_time,f_hour,year,month))
static_inputs = pickle.load(open('../2_model_training/static_inputs.pkl','rb'))
lat = gfs_data['lat']
lon = gfs_data['lon']
lat2d = static_inputs['lat']
lon2d = static_inputs['lon']
terrain = static_inputs['terrain']

times = gfs_data['time']
levels = gfs_data['levels']

cape = gfs_data['cape']
reflectivity = gfs_data['reflectivity']
lifted_index = gfs_data['lifted_idx']
precip_rate = gfs_data['precip_rate']
# binary_ltg = gfs_data['binary_ltg']
# fed = gfs_data['fed']

for i,time in enumerate(times):
    if i>=0:
        dt64 = times.values[i]
        dt_temp = pd.to_datetime(dt64)
        time_str = dt_temp.strftime('%Y-%m-%d %H:%M:%S')
        time_str1 = dt_temp.strftime('%Y%m%d%H%M')
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        cb = ax.contourf(lon,lat,cape.values[i,:,:],transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
        ax.add_feature(cfeature.STATES,edgecolor="gray")
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='white', alpha=0.5, linestyle='--')
        plt.suptitle('CAPE (J/kg): '+time_str)
        save_dir = './images/CAPE/%s/'%(f_hour)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname='CAPE_%s.%s.%s.png'%(time_str1,init_time,f_hour)
        plt.savefig(save_dir+fname)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        cb = ax.contourf(lon,lat,reflectivity.values[i,:,:],transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
        ax.add_feature(cfeature.STATES,edgecolor="gray")
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='white', alpha=0.5, linestyle='--')
        plt.suptitle('Reflectivity (dB): '+time_str)
        save_dir = './images/reflectivity/%s/'%(f_hour)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname='Z_%s.%s.%s.png'%(time_str1,init_time,f_hour)
        plt.savefig(save_dir+fname)
        plt.close()

        fig  = plt.figure()
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        cb = ax.contourf(lon,lat,lifted_index.values[i,:,:],transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
        ax.add_feature(cfeature.STATES,edgecolor="gray")
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='white', alpha=0.5, linestyle='--')
        plt.suptitle('Lifted Index (K): '+time_str)
        save_dir = './images/LI/%s/'%(f_hour)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname='LI_%s.%s.%s.png'%(time_str1,init_time,f_hour)
        plt.savefig(save_dir+fname)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        cb = ax.contourf(lon,lat,precip_rate.values[i,:,:],transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
        ax.add_feature(cfeature.STATES,edgecolor="gray")
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='white', alpha=0.5, linestyle='--')
        plt.suptitle('Precip Rate: '+time_str)
        save_dir = './images/Precip_Rate/%s/'%(f_hour)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname='Precip_Rate_%s.%s.%s.png'%(time_str1,init_time,f_hour)
        plt.savefig(save_dir+fname)
        plt.close()

        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        # cb = ax.contourf(lon,lat,binary_ltg.values[i,:,:],transform=ccrs.PlateCarree())
        # ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
        # ax.add_feature(cfeature.STATES,edgecolor="gray")
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='white', alpha=0.5, linestyle='--')
        # plt.suptitle('Binary Lightning: '+time_str)
        # save_dir = './images/binary_ltg/%s/'%(f_hour)
        # if not os.path.isdir(save_dir):
        #     os.makedirs(save_dir)
        # fname='Binary_Ltg_%s.%s.%s.png'%(time_str1,init_time,f_hour)
        # plt.savefig(save_dir+fname)
        # plt.close()

        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        # cb = ax.contourf(lon,lat,fed.values[i,:,:],transform=ccrs.PlateCarree())
        # ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
        # ax.add_feature(cfeature.STATES,edgecolor="gray")
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='white', alpha=0.5, linestyle='--')
        # plt.suptitle('FED: '+time_str)
        # save_dir = './images/fed/%s/'%(f_hour)
        # if not os.path.isdir(save_dir):
        #     os.makedirs(save_dir)
        # fname='FED_%s.%s.%s.png'%(time_str1,init_time,f_hour)
        # plt.savefig(save_dir+fname)
        # plt.close()

w = gfs_data['w']
graupel_q = gfs_data['graupel_q']
ice_q = gfs_data['ice_q']
rain_q = gfs_data['rain_q']
snow_q = gfs_data['snow_q']

for t,time in enumerate(times):
    if t>=0:
        fig  = plt.figure()
        dt64 = times.values[t]
        dt_temp = pd.to_datetime(dt64)
        time_str = dt_temp.strftime('%Y-%m-%d %H:%M:%S')
        time_str1 = dt_temp.strftime('%Y%m%d%H%M')

        for l in range(19):
            ax = fig.add_subplot(4,5,l+1, projection=ccrs.PlateCarree())
            cb = ax.contourf(lon,lat,graupel_q.values[t,:,:,l],transform=ccrs.PlateCarree())
            ax.set_title(str(levels[l].values)+' mb')
            ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
            ax.add_feature(cfeature.STATES,edgecolor="gray")
        plt.suptitle('Graupel mixing ratio: '+time_str)
        save_dir = './images/graupel_q/%s/'%(f_hour)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname = 'graupel_q_%s.png'%(time_str1)
        plt.savefig(save_dir+fname)
        plt.close()

for t,time in enumerate(times):
    if t>=0:
        fig  = plt.figure()
        dt64 = times.values[t]
        dt_temp = pd.to_datetime(dt64)
        time_str = dt_temp.strftime('%Y-%m-%d %H:%M:%S')
        time_str1 = dt_temp.strftime('%Y%m%d%H%M')

        for l in range(19):
            ax = fig.add_subplot(4,5,l+1, projection=ccrs.PlateCarree())
            cb = ax.contourf(lon,lat,ice_q.values[t,:,:,l],transform=ccrs.PlateCarree())
            ax.set_title(str(levels[l].values)+' mb')
            ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
            ax.add_feature(cfeature.STATES,edgecolor="gray")
            
        plt.suptitle('Ice mixing ratio: '+time_str)
        save_dir = './images/ice_q/%s/'%(f_hour)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname = 'ice_q_%s.png'%(time_str1)
        plt.savefig(save_dir+fname)
        plt.close()


for t,time in enumerate(times):
    if t>=0:
        fig  = plt.figure()
        dt64 = times.values[t]
        dt_temp = pd.to_datetime(dt64)
        time_str = dt_temp.strftime('%Y-%m-%d %H:%M:%S')
        time_str1 = dt_temp.strftime('%Y%m%d%H%M')

        for l in range(19):
            ax = fig.add_subplot(4,5,l+1, projection=ccrs.PlateCarree())
            cb = ax.contourf(lon,lat,snow_q.values[t,:,:,l],transform=ccrs.PlateCarree())
            ax.set_title(str(levels[l].values)+' mb')
            ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
            ax.add_feature(cfeature.STATES,edgecolor="gray")
        plt.suptitle('Snow mixing ratio: '+time_str)
        save_dir = './images/snow_q/%s/'%(f_hour)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname = 'snow_q_%s.png'%(time_str1)
        plt.savefig(save_dir+fname)
        plt.close()

for t,time in enumerate(times):
    if t>=0:
        fig  = plt.figure()
        dt64 = times.values[t]
        dt_temp = pd.to_datetime(dt64)
        time_str = dt_temp.strftime('%Y-%m-%d %H:%M:%S')
        time_str1 = dt_temp.strftime('%Y%m%d%H%M')

        for l in range(19):
            ax = fig.add_subplot(4,5,l+1, projection=ccrs.PlateCarree())
            cb = ax.contourf(lon,lat,rain_q.values[t,:,:,l],transform=ccrs.PlateCarree())
            ax.set_title(str(levels[l].values)+' mb')
            ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
            ax.add_feature(cfeature.STATES,edgecolor="gray")
            
        plt.suptitle('Rain mixing ratio: '+time_str)
        save_dir = './images/rain_q/%s/'%(f_hour)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname = 'rain_q_%s.png'%(time_str1)
        plt.savefig(save_dir+fname)
        plt.close()
    

for t,time in enumerate(times):
    if t>=0:
        fig  = plt.figure()
        dt64 = times.values[t]
        dt_temp = pd.to_datetime(dt64)
        time_str = dt_temp.strftime('%Y-%m-%d %H:%M:%S')
        time_str1 = dt_temp.strftime('%Y%m%d%H%M')

        for l in range(19):
            ax = fig.add_subplot(4,5,l+1, projection=ccrs.PlateCarree())
            cb = ax.contourf(lon,lat,w.values[t,:,:,l],transform=ccrs.PlateCarree())
            ax.set_title(str(levels[l].values)+' mb')
            ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
            ax.add_feature(cfeature.STATES,edgecolor="gray")
            
        plt.suptitle('Vertical Velocity: '+time_str)
        save_dir = './images/w/%s/'%(f_hour)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname = 'vert_vel_%s.png'%(time_str1)
        plt.savefig(save_dir+fname)
        plt.close()