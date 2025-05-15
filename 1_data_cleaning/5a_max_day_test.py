import xarray as xr
import numpy as np
import os


def main():
    var_list = ['cape','lifted_idx','reflectivity','precip_rate',
                'rain_q','ice_q','snow_q','graupel_q','w']
    data_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/5_vars_max_day/'+var_list[0]+'/'
    files = sorted(os.listdir(data_dir))

    for file in files:
        print(file)
        ds = xr.open_dataset(data_dir+file,engine='netcdf4')
        print(ds)

if __name__=="__main__":
    main()