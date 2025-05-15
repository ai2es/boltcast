import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pickle
import pandas as pd

mpl.rcParams['axes.labelsize'] = 18 #fontsize in pts
mpl.rcParams['axes.titlesize'] = 18 
mpl.rcParams['xtick.labelsize'] = 12 
mpl.rcParams['ytick.labelsize'] = 12 
mpl.rcParams['legend.fontsize'] = 18 

def load_the_data(rotation=4,model_initialization_time = '12/06/2022 06:00:00'):
    from BC_analysis_data_loader import load_test_data_nc
    #load the non-normalized data
    test_ds = load_test_data_nc('/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/binary/',
                            rotation=4)
    uno_ds = test_ds.sel(valid_times=model_initialization_time)
    del test_ds
    lat = uno_ds['lat'].values
    lon = uno_ds['lon'].values
    features = uno_ds['features'].values
    x = uno_ds['x'].values
    y = uno_ds['y'].values
    y = np.float32(uno_ds['y'].values)
    temp_dict = {'x':x,'y':y,'lat':lat,'lon':lon,'features':features}
    pickle.dump(temp_dict,open('./input_output_dict.pkl','wb'))
    
def make_the_images():
    temp_dict = pickle.load(open('./input_output_dict.pkl','rb'))
    x = temp_dict['x']
    y = temp_dict['y']
    features = temp_dict['features']
    lat = temp_dict['lat']
    lon = temp_dict['lon']

    for d in [0,1,2,3]:
        for f,feature in enumerate(features):
            if f>=0:
                print(f,feature)
                fig, axes = plt.subplots(figsize=(10,6),
                            nrows=1,
                            ncols=1,
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            layout='constrained')
                temp_x = x[d,:,:,f]
                axes.pcolormesh(lon,lat,temp_x,cmap='viridis')
                axes.add_feature(cfeature.STATES,edgecolor='white')
                axes.add_feature(cfeature.COASTLINE,edgecolor='white')
                plt.savefig('./in_n_out/%s_day_%s.png'%(feature,d))
                plt.close()

    for d in [0,1,2,3]:
        print(y.shape)
        fig, axes = plt.subplots(figsize=(10,6),
                            nrows=1,
                            ncols=1,
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            layout='constrained')
        temp_y = y[:,:,d]
        axes.pcolormesh(lon,lat,temp_y,cmap='Greys')
        axes.add_feature(cfeature.STATES,edgecolor='black')
        axes.add_feature(cfeature.COASTLINE,edgecolor='black')
        plt.savefig('./in_n_out/%s_day_%s.png'%('labels',d))
        plt.close()


def main():
    load_the_data(rotation=4,model_initialization_time = '06/13/2022 12:00:00')
    make_the_images()

if __name__ == "__main__":
    main()
