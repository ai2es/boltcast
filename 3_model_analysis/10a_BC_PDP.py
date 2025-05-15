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

    d=0 #day index
    print('loading the test dataset')
    test_ds = load_test_data_nc('/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/binary/',
                            rotation=4)

    print('extracting the data')
    x = test_ds['x'].values
    y = test_ds['y'].values
    y = np.float32(test_ds['y'].values)
    y = np.swapaxes(y,1,3)
    y = np.swapaxes(y,2,3)

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

    print('loading the model')
    if model_type=='UNet':
        model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/UNet/models/'
        model_fname = 'BC_UNet_rot_4_LR_0.000010000_deep_3_nconv_3_conv_size_4_stride_1_epochs_500__binary_batch_32_symmetric__SD_0.0_conv_relu__last_sigmoid__model.keras'
    if model_type=='LSTM':
        model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/models/'
        model_fname = 'BC_LSTM_rot_4_conv_4_conv_deep_%s_lstm_deep_%s_no_drop_no_shuffle_model.keras'%(conv_deep,lstm_deep)
    model = tf.keras.models.load_model(model_dir+model_fname)

    for f,feature in enumerate(features):
        #get the features across the entire test dataset
        feature_values = np.squeeze(x[:,d,lat_idx,lon_idx,f])
        feature_values = np.unique(feature_values)
        preds_array = np.zeros([len(feature_values),len(neighborhoods)])
        for n,neighborhood in enumerate(neighborhoods):

            temp_lat_idx = lat_idx-neighborhood
            temp_lat_idx2 = lat_idx+neighborhood

            temp_lon_idx = lon_idx-neighborhood
            temp_lon_idx2 = lon_idx+neighborhood
            
            for idx in range(len(feature_values)):
                if idx%10==0:
                    print(idx,len(feature_values))
                x_one_temp = x_one
                x_one_temp[d,temp_lat_idx:temp_lat_idx2,temp_lon_idx:temp_lon_idx2,f]=feature_values[idx]
                x_one_norm = min_max_scale_uno_datetime(x_one_temp) 
                x_pred = np.expand_dims(x_one_norm,axis=0)
                y_pred_uno = model.predict(x=x_pred,verbose=0)
                y_pred_uno = np.squeeze(np.squeeze(y_pred_uno))
                preds_array[idx,n] = y_pred_uno[d,lat_idx,lon_idx]
        pred_dict = {'feature_values':feature_values,'preds_array':preds_array}
        if model_type=='UNet':
            pickle.dump(pred_dict,open('./pdps/%s_%s_%s_pred_dict.pkl'%(loc,model_type,feature),'wb'))
        if model_type=='LSTM':
            pickle.dump(pred_dict,open('./pdps/%s_%s_%s_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(loc,model_type,feature,conv_deep,lstm_deep),'wb'))
        del pred_dict, preds_array, y_pred_uno, x_pred, x_one_norm, x_one_temp

if __name__=='__main__':
    calc_pdp_pkl(loc='vance',model_type='UNet')
    calc_pdp_pkl(loc='vance',model_type='LSTM')
    calc_pdp_pkl(loc='wright_patt',model_type='UNet')
    calc_pdp_pkl(loc='wright_patt',model_type='LSTM')



