import pygrib
import os
import shutil
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import matplotlib.colors as mcolors

def get_HREF_48(yr='2022',mo='06',day='12'):

    href_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/12_SPC_HREF/hrefct_full_%s/%s%s%s/thunder/'%(yr,yr,mo,day)
    files = sorted(os.listdir(href_dir))
    label = 'Thunderstorm probability'

    print()
    for f,file in enumerate(files):
        if f==72: #Day 2, 00Z
            grbs = pygrib.open(href_dir+file)
            grbindx = pygrib.index(href_dir+file,'name','typeOfLevel','level')
            data = grbindx.select(name=label,typeOfLevel='surface',level=0)[0].values
            for grb in grbs:
                projection_params = grb.projparams
                print(projection_params)
                proj_a = projection_params['a']
                proj_b = projection_params['b']
                lon_0 = projection_params['lon_0']
                lat_0 = projection_params['lat_0']
            hrrr_proj = ccrs.LambertConformal(central_longitude=lon_0, 
                                        central_latitude=lat_0,
                                        globe=ccrs.Globe(semimajor_axis=proj_a,
                                                            semiminor_axis=proj_b))
            lat,lon = grb.latlons()
            return hrrr_proj,lat,lon,data

def get_HREF_24(yr='2022',mo='06',day='13'):

    href_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/12_SPC_HREF/hrefct_full_%s/%s%s%s/thunder/'%(yr,yr,mo,day)
    files = sorted(os.listdir(href_dir))
    label = 'Thunderstorm probability'
    print()
    for f,file in enumerate(files):
        if f==48: #Day 1, 12Z
            grbs = pygrib.open(href_dir+file)
            grbindx = pygrib.index(href_dir+file,'name','typeOfLevel','level')
            data = grbindx.select(name=label,typeOfLevel='surface',level=0)[0].values
            for grb in grbs:
                print(f,file)
                print(grb)
                projection_params = grb.projparams
                print(projection_params)
                proj_a = projection_params['a']
                proj_b = projection_params['b']
                lon_0 = projection_params['lon_0']
                lat_0 = projection_params['lat_0']
                hrrr_proj = ccrs.LambertConformal(central_longitude=lon_0, 
                                        central_latitude=lat_0,
                                        globe=ccrs.Globe(semimajor_axis=proj_a,
                                                            semiminor_axis=proj_b))
                lat,lon = grb.latlons()
                return hrrr_proj,lat,lon,data
if __name__=='__main__':
    get_HREF_24()