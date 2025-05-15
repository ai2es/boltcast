import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

def main():
    print("building 30dbz output thresholds")
    load_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/9_ds/binary_ltg/'
    init_time = '00Z'
    load_file = init_time+'_x.nc'
    ds = xr.open_dataset(load_dir+load_file,engine='netcdf4')
    valid_times = ds['valid_times'].values
    x = ds['x'].values
    reflectivity = x[:,:,:,:,2] #index 2 is reflectivity
    del ds, x
    
    ts = pd.Timestamp('06/17/2023 00:00')
    time_idx = np.where(valid_times==ts)[0]
    print(time_idx)
    single_refl_output = np.squeeze(reflectivity[time_idx,:,:,:])
    single_refl_output = single_refl_output>=30

    fig, axs = plt.subplots(nrows=1,ncols=4)
    cb = axs[3].imshow(single_refl_output[0,:,:],cmap='coolwarm',vmin=0,vmax=1)#day1
    cb = axs[2].imshow(single_refl_output[1,:,:],cmap='coolwarm',vmin=0,vmax=1)#day2
    cb = axs[1].imshow(single_refl_output[2,:,:],cmap='coolwarm',vmin=0,vmax=1)#day3
    cb = axs[0].imshow(single_refl_output[3,:,:],cmap='coolwarm',vmin=0,vmax=1)#day4
    plt.savefig('test.png')
    plt.close()



if __name__=="__main__":
    main()