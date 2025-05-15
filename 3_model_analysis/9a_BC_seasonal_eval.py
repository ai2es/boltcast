import tensorflow as tf
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from BC_analysis_data_loader import *
import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, auc
import os

def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")

def parse_args():
    parser = argparse.ArgumentParser(description='BoltCast_metrics_seasonal', fromfile_prefix_chars='@')
    parser.add_argument('--model2eval',type=str,default='UNet')
    parser.add_argument('--rotation',type=int,default=4)
    parser.add_argument('--lrate',type=float,default=1e-5)
    args = parser.parse_args()
    return args

def metrics(X_norm, y, model,args):

    day_1_labels = y[:,0,:,:]
    day_2_labels = y[:,1,:,:]
    day_3_labels = y[:,2,:,:]
    day_4_labels = y[:,3,:,:]

    thresh=.25
    print('generating the metrics list')
    opt = keras.optimizers.Adam(learning_rate=args.lrate, amsgrad=False)
    loss_tf = tf.keras.losses.BinaryCrossentropy()
    auc_roc_tf = tf.keras.metrics.AUC(name='auc_ROC',curve='ROC')
    auc_pr_tf = tf.keras.metrics.AUC(name='auc_PR',curve='PR')
    acc_tf = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy',threshold=thresh)
    prec_tf = tf.keras.metrics.Precision(name='precision',thresholds=thresh)
    recall_tf = tf.keras.metrics.Recall(name='recall',thresholds=thresh)
    all_metrics = [auc_roc_tf,auc_pr_tf,acc_tf,prec_tf,recall_tf]
    
    #compile the model with the new metrics
    model.compile(optimizer=opt,loss=loss_tf,metrics=all_metrics)
    print('compiling the model')
    print(model)

    #predict the model for the given seasonal data
    model_output = model.predict(X_norm)

    #evaluate the entire model output for the given metrics
    eval_dict = model.evaluate(X_norm,y,return_dict=True,verbose=0)

    #evaluate the each day's performance
    day_1_dict = daily_metrics(day_1_labels,model_output[:,0,:,:])
    day_2_dict = daily_metrics(day_2_labels,model_output[:,1,:,:])
    day_3_dict = daily_metrics(day_3_labels,model_output[:,2,:,:])
    day_4_dict = daily_metrics(day_4_labels,model_output[:,3,:,:])

    all_metrics_dict = {
        'eval_dict':eval_dict,
        'day_1_dict':day_1_dict,
        'day_2_dict':day_2_dict,
        'day_3_dict':day_3_dict,
        'day_4_dict':day_4_dict
    }
    return all_metrics_dict

def daily_metrics(labels,model_output):

    #set the thesholds
    thresh = np.arange(0.05,1.05,0.05)
    
    #statistics we need for performance diagram 
    tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())#a
    fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())#b
    fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())#c
    tn = tf.keras.metrics.TrueNegatives(thresholds=thresh.tolist())#d

    day_tp = tp(labels,model_output)
    day_fp = fp(labels,model_output)
    day_fn = fn(labels,model_output)
    day_tn = tn(labels,model_output)

    day_pod = day_tp/(day_tp+day_fn)
    day_srs = day_tp/(day_tp+day_fp)
    day_csi = day_tp/(day_tp+day_fn+day_fp)

    day_precision, day_recall, day_thresholds = precision_recall_curve(np.ravel(labels),np.ravel(model_output))
    day_auc = auc(day_recall,day_precision)
    day_dict = {
        'day_csi':day_csi, 
        'day_srs':day_srs,
        'day_pod':day_pod,
        'day_auc':day_auc,
        'day_precision': day_precision,
        'day_recall':day_recall
        }
    return day_dict

def calc_metrics(args):
    print('calculating the seasonal aucs and csis')
    rot_dict = build_rotations()
    
    #2=test, 1=val, 0=train
    ds_type=2

    seasonal_ds_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/binary/'
    unet_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/UNet/models/'
    lstm_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/models/'

    unet_model_fname = 'BC_UNet_rot_%s_LR_0.000010000_deep_3_nconv_3_conv_size_4_stride_1_epochs_500__binary_batch_32_symmetric__SD_0.0_conv_relu__last_sigmoid__model.keras'%(args.rotation)
    lstm_model_fname = 'BC_LSTM_rot_%s_conv_4_conv_deep_0_lstm_deep_1_no_drop_no_shuffle_model.keras'%(args.rotation)
    
    print('loading the models, rotation:',args.rotation)
    print(unet_model_fname)
    print(lstm_model_fname)
    unet_model = tf.keras.models.load_model(unet_dir+unet_model_fname)
    print('unet',unet_model)
    lstm_model = tf.keras.models.load_model(lstm_dir+lstm_model_fname)
    print('lstm',lstm_model)

    for season in ['summer','fall','winter','spring']:
        #get the data for the specific season
        rot_files = rot_dict[args.rotation][ds_type]
        if args.rotation==0:
            apdx=4
        else: 
            apdx=args.rotation-1
        print('loading the data for: %s rotation: %s'%(season,args.rotation))
        fload = '%s_%s.nc'%(season,apdx)
        ds = xr.open_dataset(seasonal_ds_dir+fload,engine='netcdf4')
        X = ds['x'].values
        X_norm = min_max_scale(ds)
        y = ds['y'].values
        y = np.swapaxes(y,1,3)
        y = np.swapaxes(y,2,3)
        print(season,'evaluating the unet, rotation:',args.rotation)
        unet_metrics = metrics(model=unet_model,X_norm=X_norm,y=y,args=args)
        print(season,'evaluating the lstm, rotation:',args.rotation)
        lstm_metrics = metrics(model=lstm_model,X_norm=X_norm,y=y,args=args)
        metrics_dict = {
            'unet':unet_metrics,
            'lstm':lstm_metrics
        }
        res_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/BoltCast_seasonal_metrics/'
        if os.path.isdir(res_dir)==False:
            os.makedirs(res_dir)
        pickle.dump(metrics_dict,open(res_dir+'%s_rot_%s_metrics.pkl'%(season,args.rotation),'wb'))
        #print(metrics_dict)

def main():
    extract_slurm_env()
    args = parse_args()
    calc_metrics(args=args)

if __name__=='__main__':
    main()