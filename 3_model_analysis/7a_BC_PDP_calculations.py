import numpy as np
import xarray as xr
import os
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from BC_analysis_data_loader import load_test_data_nc, min_max_scale, min_max_scale_uno_datetime
import tensorflow as tf
import pickle
import copy
import time

def calc_pdp_pkl(loc='vance',
                model_type='UNet',
                lstm_deep=1,
                conv_deep=0):

    print('calculating the pdp:',loc,model_type)
    print('declaring the lat lons of interest for the partial dependence plots')
    if loc=='wright_patt':
        loc_latlon = [39.819527, -84.067406+360]
        date = '06/13/2022 12:00:00.000000000'
        title_text = 'Wright Patterson AFB, 06/13/2022 12Z'

    if loc=='vance':
        loc_latlon = [36.3393, -97.9131+360]
        date = '12/09/2022 12:00:00.000000000'
        title_text = 'Vance AFB, 12/09/2022 12Z'

    neighborhoods = [0,4,8,12]

    
    print('loading the test dataset')
    test_ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/BC_edits_v2/BC_test_data_nc/rot_4_test.nc',engine='netcdf4')

    print('extracting the data')
    x = test_ds['x'].values

    lat = test_ds['lat'].values
    lon = test_ds['lon'].values

    print('finding the lat lon of the afb in the gfs grid')
    lat_idx = (np.abs(lat - loc_latlon[0])).argmin()
    lon_idx = (np.abs(lon - loc_latlon[1])).argmin()

    features = test_ds['features'].values
    days = test_ds['days'].values

    print('get the case study data')
    case_data = test_ds.sel(valid_times=date)
    x_one = case_data['x'].values
    print('x_one.shape:',x_one.shape)
    del test_ds, case_data

    print('loading the model')
    model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AIES_reviews_v2/'
    model_fname = 'model.keras'
    if model_type=='UNet':
        exp_name = 'BC_UNet_rot_4_lrate_0.000010000_/'
    if model_type=='LSTM':
        exp_name = 'BC_LSTM_rot_4_lrate_0.000010000_/'
    model = tf.keras.models.load_model(model_dir+exp_name+model_fname)

    for f,feature in enumerate(features):
        
        #get the features across the entire test dataset, for day d=0
        d=0
        feature_values = np.squeeze(x[:,d,lat_idx,lon_idx,f])
        feature_values = np.unique(feature_values)
        preds_array = np.zeros((4,len(feature_values),len(neighborhoods)))#day, feature, neighborhood size

        for n,neighborhood in enumerate(neighborhoods):

            temp_lat_idx = lat_idx-neighborhood
            temp_lat_idx2 = lat_idx+neighborhood

            temp_lon_idx = lon_idx-neighborhood
            temp_lon_idx2 = lon_idx+neighborhood
            
            for idx in range(len(feature_values)):
                if idx%10==0:
                    print(idx,len(feature_values))
                x_one_temp = copy.deepcopy(x_one)
                x_one_temp[d,temp_lat_idx:temp_lat_idx2,temp_lon_idx:temp_lon_idx2,f]=feature_values[idx]
                x_one_norm = min_max_scale_uno_datetime(x_one_temp)
                x_pred = np.expand_dims(x_one_norm,axis=0)
                y_pred_uno = model.predict(x=x_pred,verbose=0)
                y_pred_uno = np.squeeze(np.squeeze(y_pred_uno))
                for dd in range(4):#day index
                    preds_array[dd,idx,n] = y_pred_uno[dd,lat_idx,lon_idx]
                del y_pred_uno, x_pred, x_one_norm, x_one_temp

        pred_dict = {'feature_values':feature_values,'preds_array':preds_array}
        pickle.dump(pred_dict,open('./AIES_results_reviewer_edits_v2/pdp/%s_%s_%s_pred_dict.pkl'%(loc,model_type,feature),'wb'))
        del pred_dict, preds_array
    del x_one

if __name__=='__main__':

    print('starting the PDP clock')
    start_time = time.time()
    calc_pdp_pkl(loc='vance',model_type='UNet')
    calc_pdp_pkl(loc='vance',model_type='LSTM')
    calc_pdp_pkl(loc='wright_patt',model_type='UNet')
    calc_pdp_pkl(loc='wright_patt',model_type='LSTM')
    end_time = time.time()
    print(f"The PDP Calculations Took {end_time - start_time:.2f} seconds to run.")
    print('END of 10a_BC_PDP_calculations.py')