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
import matplotlib

def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")

    return slurm_vars

def parse_args():
    parser = argparse.ArgumentParser(description='BoltCast_metrics_seasonal', fromfile_prefix_chars='@')
    parser.add_argument('--rotation',type=int,default=0)
    parser.add_argument('--lrate',type=float,default=0.00001)
    args = parser.parse_args()
    return args

def metrics(X_norm, y, model,args,season,model2eval):

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

    #predict the model for the given seasonal data
    model_output = model.predict(X_norm)

    #evaluate the entire model output for the given metrics
    eval_dict = model.evaluate(X_norm,y,return_dict=True,verbose=0)

    #evaluate the each day's performance
    day_1_dict = daily_metrics_fn(day_1_labels,model_output[:,0,:,:])
    day_2_dict = daily_metrics_fn(day_2_labels,model_output[:,1,:,:])
    day_3_dict = daily_metrics_fn(day_3_labels,model_output[:,2,:,:])
    day_4_dict = daily_metrics_fn(day_4_labels,model_output[:,3,:,:])

    all_metrics_dict = {
        'eval_dict':eval_dict,
        'day_1_dict':day_1_dict,
        'day_2_dict':day_2_dict,
        'day_3_dict':day_3_dict,
        'day_4_dict':day_4_dict
    }

    seasonal_output = {'season':season,'rotation':args.rotation,'model_type':model2eval,'model_output':model_output,'y_true':y,'all_metrics_dict':all_metrics_dict}
    save_dir = '/scratch/bmac87/BC_reviewer_analysis_v2/'
    fname_pkl = '%s_rot_%s_season_%s_output.pkl'%(model2eval,args.rotation,season)
    pickle.dump(seasonal_output,open(save_dir+fname_pkl,'wb'))

def daily_metrics_fn(labels,model_output):

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

def calc_metrics_per_rotation(args):
    print('calculating the seasonal aucs and csis, rotation:',args.rotation)
    rot_dict = build_rotations()
    
    #2=test, 1=val, 0=train
    ds_type=2

    seasonal_ds_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/binary/'
    model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AIES_reviews_v2/'

    unet_exp_name = 'BC_UNet_rot_%s_lrate_0.000010000_/'%(args.rotation)
    lstm_exp_name = 'BC_LSTM_rot_%s_lrate_0.000010000_/'%(args.rotation)
    model_fname = 'model.keras'

    print('loading the models, rotation:',args.rotation)
    print(unet_exp_name)
    print(lstm_exp_name)

    unet_model = tf.keras.models.load_model(model_dir+unet_exp_name+model_fname)
    print('unet',unet_model)
    print('model loaded successfully')

    lstm_model = tf.keras.models.load_model(model_dir+unet_exp_name+model_fname)
    print('lstm',lstm_model)
    print('model loaded successfully')


    for season in ['summer','fall','winter','spring']:
        #get the test data for the specific season
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
        metrics(model=unet_model,X_norm=X_norm,y=y,args=args,season=season,model2eval='UNet')

        print(season,'evaluating the lstm, rotation:',args.rotation)
        metrics(model=lstm_model,X_norm=X_norm,y=y,args=args,season=season,model2eval='LSTM')
        del y, X, X_norm, ds, fload

def calc_metrics_per_season_all_rotations(model):
    rotations = [0,1,2,3,4]
    load_dir = '/scratch/bmac87/BC_reviewer_analysis_v2/'
    seasons = ['summer','fall','winter','spring']
    season_dict = {}
    for season in seasons:
        y_true_list = []
        y_pred_list = []
        for rot in rotations:
            fname = '%s_rot_%s_season_%s_output.pkl'%(model,rot,season)
            data = pickle.load(open(load_dir+fname,'rb'))
            y_true_list.append(data['y_true'])
            y_pred_list.append(data['model_output'])
            del data, fname
        y_true_season_all = np.concatenate(y_true_list,axis=0)
        y_pred_season_all = np.squeeze(np.concatenate(y_pred_list,axis=0))
        del y_true_list, y_pred_list

        daily_mets = {}
        for day in range(4):
            daily_dict = daily_metrics_fn(labels=y_true_season_all[:,day,:,:],model_output=y_pred_season_all[:,day,:,:])
            print(daily_dict)
            daily_mets.update({'Day %s'%(day+1):daily_dict})
            del daily_dict
        season_dict.update({season:daily_mets})
        del daily_mets, y_true_season_all, y_pred_season_all
    pickle.dump(season_dict,open(load_dir+'%s_season_stats_bar.pkl'%model,'wb'))

def generate_plot(plot_metric='day_csi'):
    print('generating the plot')
    results_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/BoltCast_seasonal_metrics/'
    rotations = ['0','1','2','3','4']
    seasons = ['summer','fall','winter','spring']

    if plot_metric=='day_csi':
        ylims = [.14,.5]
        ytick_nums = [.14,.16,.18,.2,.22,.24,.26,.28,.30,.32,.34,.36,.38,.4,.42,.44,.46,.48,.50]
        ytick_strs = ['0.14','0.16','0.18','0.20','0.22','0.24','0.26','0.28','0.30','0.32','0.34','0.36','0.38','0.40','0.42','0.44','0.46','0.48','0.50']
        ylabel = 'Max. CSI'
        y_text = .48
        sn_text = .485

        summer_text = .325
        fall_text = .225
        winter_text = .165
        spring_text  = .245

    
    if plot_metric=='day_auc':
        ylims = [.2,.66]
        ytick_nums = [.2,.24,.28,.32,.36,.4,.44,.48,.52,.56,.60,.64,.68,.72]
        ytick_strs = ['0.20','0.24','0.28','0.32','0.36','0.40','0.44','0.48','0.52','0.56','0.60','0.64','0.68','0.72']
        ylabel = 'AUC'
        y_text = .655
        sn_text = .68

        summer_text = .425
        fall_text = .305
        winter_text = .205
        spring_text  = .325


    unet_csis = np.zeros((4,5,4))#season, rotations, day
    lstm_csis = np.zeros((4,5,4))#season, rotations, day

    for s,season in enumerate(seasons):
        for r,rot in enumerate(rotations):
            print('loading results for season and rotation:',season, rot)
            results_file = '%s_rot_%s_metrics.pkl'%(season,rot)
            results = pickle.load(open(results_dir+results_file,'rb'))
            unet_results = results['unet']
            lstm_results = results['lstm']

            unet_csis[s,r,0] = np.max(unet_results['day_1_dict'][plot_metric])
            unet_csis[s,r,1] = np.max(unet_results['day_2_dict'][plot_metric])
            unet_csis[s,r,2] = np.max(unet_results['day_3_dict'][plot_metric])
            unet_csis[s,r,3] = np.max(unet_results['day_4_dict'][plot_metric])

            lstm_csis[s,r,0] = np.max(lstm_results['day_1_dict'][plot_metric])
            lstm_csis[s,r,1] = np.max(lstm_results['day_2_dict'][plot_metric])
            lstm_csis[s,r,2] = np.max(lstm_results['day_3_dict'][plot_metric])
            lstm_csis[s,r,3] = np.max(lstm_results['day_4_dict'][plot_metric])
    
    matplotlib.rcParams['axes.facecolor'] = [0.95,0.95,0.95] 

    fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(20,20))
    colors_hex_unet = ['#ca0020','#f4a582','#92c5de','#0571b0']
    colors_hex_lstm = ['#ca0020','#f4a582','#92c5de','#0571b0']
    width = 1
    xticks = np.arange(24)
    whiskerprops = dict(linewidth=3,color='black')
    capprops = dict(linewidth=3,color='black')
    medianprops = dict(linewidth=3,color='black')
    boxprops = dict(linewidth=3,color='black')
    
    axes[0].grid(axis='y')
    axes[0].set_axisbelow(True)
    axes[0].text(1.5,sn_text,'Summer',fontsize=24)
    # axes[0].bar(x=1,height=np.median(unet_csis[0,:,0]),width=width,color=colors_hex_unet[0],alpha=.6)
    axes[0].text(.6,summer_text,f"{np.median(unet_csis[0,:,0]):.2f}",fontsize=18)
    # axes[0].bar(x=2,height=np.median(unet_csis[0,:,1]),width=width,color=colors_hex_unet[1],alpha=.6)
    axes[0].text(1.6,summer_text,f"{np.median(unet_csis[0,:,1]):.2f}",fontsize=18)
    # axes[0].bar(x=3,height=np.median(unet_csis[0,:,2]),width=width,color=colors_hex_unet[2],alpha=.6)
    axes[0].text(2.6,summer_text,f"{np.median(unet_csis[0,:,2]):.2f}",fontsize=18)
    # axes[0].bar(x=4,height=np.median(unet_csis[0,:,3]),width=width,color=colors_hex_unet[3],alpha=.6)
    axes[0].text(3.6,summer_text,f"{np.median(unet_csis[0,:,3]):.2f}",fontsize=18)
    bplot = axes[0].boxplot(unet_csis[0,:,:], positions=[1,2,3,4],vert=True, patch_artist=True, widths=1, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, boxprops=boxprops)
    for patch, color in zip(bplot['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
        patch.set_hatch('//')
    del bplot

    axes[0].text(7.5,sn_text,'Fall',fontsize=24)
    # axes[0].bar(x=7,height=np.median(unet_csis[1,:,0]),width=width,color=colors_hex_unet[0],alpha=.6)
    axes[0].text(6.6,fall_text,f"{np.median(unet_csis[1,:,0]):.2f}",fontsize=18)
    # axes[0].bar(x=8,height=np.median(unet_csis[1,:,1]),width=width,color=colors_hex_unet[1],alpha=.6)
    axes[0].text(7.6,fall_text,f"{np.median(unet_csis[1,:,1]):.2f}",fontsize=18)
    # axes[0].bar(x=9,height=np.median(unet_csis[1,:,2]),width=width,color=colors_hex_unet[2],alpha=.6)
    axes[0].text(8.6,fall_text,f"{np.median(unet_csis[1,:,2]):.2f}",fontsize=18)
    # axes[0].bar(x=10,height=np.median(unet_csis[1,:,3]),width=width,color=colors_hex_unet[3],alpha=.6)
    axes[0].text(9.6,fall_text,f"{np.median(unet_csis[1,:,3]):.2f}",fontsize=18)
    bplot = axes[0].boxplot(unet_csis[1,:,:], positions=[7,8,9,10],vert=True, patch_artist=True, widths=1, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops,boxprops=boxprops)
    for patch, color in zip(bplot['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
        patch.set_hatch('//')
    del bplot

    axes[0].text(13.5,sn_text,'Winter',fontsize=24)
    # axes[0].bar(x=13,height=np.median(unet_csis[2,:,0]),width=width,color=colors_hex_unet[0],alpha=.6)
    axes[0].text(12.6,winter_text,f"{np.median(unet_csis[2,:,0]):.2f}",fontsize=18)
    # axes[0].bar(x=14,height=np.median(unet_csis[2,:,1]),width=width,color=colors_hex_unet[1],alpha=.6)
    axes[0].text(13.6,winter_text,f"{np.median(unet_csis[2,:,1]):.2f}",fontsize=18)
    # axes[0].bar(x=15,height=np.median(unet_csis[2,:,2]),width=width,color=colors_hex_unet[2],alpha=.6)
    axes[0].text(14.6,winter_text,f"{np.median(unet_csis[2,:,2]):.2f}",fontsize=18)
    # axes[0].bar(x=16,height=np.median(unet_csis[2,:,3]),width=width,color=colors_hex_unet[3],alpha=.6) 
    axes[0].text(15.6,winter_text,f"{np.median(unet_csis[2,:,3]):.2f}",fontsize=18) 
    bplot = axes[0].boxplot(unet_csis[2,:,:], positions=[13,14,15,16],vert=True, patch_artist=True, widths=1, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops,boxprops=boxprops)
    for patch, color in zip(bplot['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
        patch.set_hatch('//')
    del bplot

    axes[0].text(19.5,sn_text,'Spring',fontsize=24)
    # axes[0].bar(x=19,height=np.median(unet_csis[3,:,0]),width=width,color=colors_hex_unet[0],alpha=.6)
    axes[0].text(18.6,spring_text,f"{np.median(unet_csis[3,:,0]):.2f}",fontsize=18)
    # axes[0].bar(x=20,height=np.median(unet_csis[3,:,1]),width=width,color=colors_hex_unet[1],alpha=.6)
    axes[0].text(19.6,spring_text,f"{np.median(unet_csis[3,:,1]):.2f}",fontsize=18)
    # axes[0].bar(x=21,height=np.median(unet_csis[3,:,2]),width=width,color=colors_hex_unet[2],alpha=.6)
    axes[0].text(20.6,spring_text,f"{np.median(unet_csis[3,:,2]):.2f}",fontsize=18)
    # axes[0].bar(x=22,height=np.median(unet_csis[3,:,3]),width=width,color=colors_hex_unet[3],alpha=.6)
    axes[0].text(21.6,spring_text,f"{np.median(unet_csis[3,:,3]):.2f}",fontsize=18)
    bplot = axes[0].boxplot(unet_csis[3,:,:], positions=[19,20,21,22],vert=True, patch_artist=True, widths=1, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops,boxprops=boxprops)
    for patch, color in zip(bplot['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
        patch.set_hatch('//')
    del bplot


    axes[0].set_ylabel(ylabel,fontsize=24)
    axes[0].set_ylim(ylims)
    axes[0].set_yticks(ytick_nums,ytick_strs,fontsize=18)
    axes[0].set_title('U-Net',fontsize=24)
    axes[0].set_xticks(xticks,['','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','' ],fontsize=18,rotation=45)
    axes[0].tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=True)
    
    axes[1].grid(axis='y')
    axes[1].set_axisbelow(True)

    axes[1].text(1.5,sn_text,'Summer',fontsize=24)
    # axes[1].bar(x=1,height=np.median(lstm_csis[0,:,0]),width=width,color=colors_hex_lstm[0],alpha=.6)
    axes[1].text(.6,summer_text,f"{np.median(lstm_csis[0,:,0]):.2f}",fontsize=18)
    # axes[1].bar(x=2,height=np.median(lstm_csis[0,:,1]),width=width,color=colors_hex_lstm[1],alpha=.6)
    axes[1].text(1.6,summer_text,f"{np.median(lstm_csis[0,:,1]):.2f}",fontsize=18)
    # axes[1].bar(x=3,height=np.median(lstm_csis[0,:,2]),width=width,color=colors_hex_lstm[2],alpha=.6)
    axes[1].text(2.6,summer_text,f"{np.median(lstm_csis[0,:,2]):.2f}",fontsize=18)
    # axes[1].bar(x=4,height=np.median(lstm_csis[0,:,3]),width=width,color=colors_hex_lstm[3],alpha=.6)
    axes[1].text(3.6,summer_text,f"{np.median(lstm_csis[0,:,3]):.2f}",fontsize=18)
    bplot = axes[1].boxplot(lstm_csis[0,:,:], positions=[1,2,3,4],vert=True, patch_artist=True, widths=1, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops,boxprops=boxprops)
    for patch, color in zip(bplot['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
        patch.set_hatch('//')
    del bplot

    axes[1].text(7.5,sn_text,'Fall',fontsize=24)
    # axes[1].bar(x=7,height=np.median(lstm_csis[1,:,0]),width=width,color=colors_hex_lstm[0],alpha=.6)
    axes[1].text(6.6,fall_text,f"{np.median(lstm_csis[1,:,0]):.2f}",fontsize=18)
    # axes[1].bar(x=8,height=np.median(lstm_csis[1,:,1]),width=width,color=colors_hex_lstm[1],alpha=.6)
    axes[1].text(7.6,fall_text,f"{np.median(lstm_csis[1,:,1]):.2f}",fontsize=18)
    # axes[1].bar(x=9,height=np.median(lstm_csis[1,:,2]),width=width,color=colors_hex_lstm[2],alpha=.6)
    axes[1].text(8.6,fall_text,f"{np.median(lstm_csis[1,:,2]):.2f}",fontsize=18)
    # axes[1].bar(x=10,height=np.median(lstm_csis[1,:,3]),width=width,color=colors_hex_lstm[3],alpha=.6)
    axes[1].text(9.6,fall_text,f"{np.mean(lstm_csis[1,:,3]):.2f}",fontsize=18)
    bplot = axes[1].boxplot(lstm_csis[1,:,:], positions=[7,8,9,10],vert=True, patch_artist=True, widths=1, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops,boxprops=boxprops)
    for patch, color in zip(bplot['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
        patch.set_hatch('//')
    del bplot

    axes[1].text(13.5,sn_text,'Winter',fontsize=24)
    # axes[1].bar(x=13,height=np.median(lstm_csis[2,:,0]),width=width,color=colors_hex_lstm[0],alpha=.6)
    axes[1].text(12.6,winter_text,f"{np.median(lstm_csis[2,:,0]):.2f}",fontsize=18)
    # axes[1].bar(x=14,height=np.median(lstm_csis[2,:,1]),width=width,color=colors_hex_lstm[1],alpha=.6)
    axes[1].text(13.6,winter_text,f"{np.median(lstm_csis[2,:,1]):.2f}",fontsize=18)
    # axes[1].bar(x=15,height=np.median(lstm_csis[2,:,2]),width=width,color=colors_hex_lstm[2],alpha=.6)
    axes[1].text(14.6,winter_text,f"{np.median(lstm_csis[2,:,2]):.2f}",fontsize=18)
    # axes[1].bar(x=16,height=np.median(lstm_csis[2,:,3]),width=width,color=colors_hex_lstm[3],alpha=.6)
    axes[1].text(15.6,winter_text,f"{np.median(lstm_csis[2,:,3]):.2f}",fontsize=18)
    bplot = axes[1].boxplot(lstm_csis[2,:,:], positions=[13,14,15,16],vert=True, patch_artist=True, widths=1, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops,boxprops=boxprops)
    for patch, color in zip(bplot['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
        patch.set_hatch('//')
    del bplot

    axes[1].text(19.5,sn_text,'Spring',fontsize=24)
    # axes[1].bar(x=19,height=np.median(lstm_csis[3,:,0]),width=width,color=colors_hex_lstm[0],alpha=.6)
    axes[1].text(18.6,spring_text,f"{np.median(lstm_csis[3,:,0]):.2f}",fontsize=18)
    # axes[1].bar(x=20,height=np.median(lstm_csis[3,:,1]),width=width,color=colors_hex_lstm[1],alpha=.6)
    axes[1].text(19.6,spring_text,f"{np.median(lstm_csis[3,:,1]):.2f}",fontsize=18)
    # axes[1].bar(x=21,height=np.median(lstm_csis[3,:,2]),width=width,color=colors_hex_lstm[2],alpha=.6)
    axes[1].text(20.6,spring_text,f"{np.median(lstm_csis[3,:,2]):.2f}",fontsize=18)
    # axes[1].bar(x=22,height=np.median(lstm_csis[3,:,3]),width=width,color=colors_hex_lstm[3],alpha=.6)
    axes[1].text(21.6,spring_text,f"{np.median(lstm_csis[3,:,3]):.2f}",fontsize=18)
    bplot = axes[1].boxplot(lstm_csis[3,:,:], positions=[19,20,21,22],vert=True, patch_artist=True, widths=1, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops,boxprops=boxprops)
    for patch, color in zip(bplot['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
        patch.set_hatch('//')
    del bplot

    axes[1].set_ylabel(ylabel,fontsize=24)
    axes[1].set_ylim(ylims)
    axes[1].set_yticks(ytick_nums,ytick_strs,fontsize=18)
    axes[1].set_title('LSTM',fontsize=24)
    axes[1].set_xticks(xticks,['','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','' ],fontsize=18,rotation=45)
    axes[1].tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=True)
    axes[0].text(.2,y_text,'(a)',fontsize = 24,fontweight = 'bold')
    axes[1].text(.2,y_text,'(b)',fontsize = 24,fontweight = 'bold')
    
    plt.savefig('./AIES_results_reviewer_edits/seasonal_box_plot_%s.pdf'%(plot_metric))
    plt.savefig('./AIES_results_reviewer_edits/seasonal_box_plot_%s.png'%(plot_metric))
    plt.close()

def generate_plot2(plot_metric='day_auc'):#for AIES reviewer comments
    seasons = ['summer','fall','winter','spring']
    unet_stats = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/BC_edits_v2/BC_reviewer_analysis_v2/UNet_season_stats_bar.pkl','rb'))
    lstm_stats = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/BC_edits_v2/BC_reviewer_analysis_v2/LSTM_season_stats_bar.pkl','rb'))

    summer_text = .425
    fall_text = .305
    winter_text = .205
    spring_text  = .325
    

    if plot_metric=='day_auc':
        ylims = [.2,.66]
        ytick_nums = [.2,.24,.28,.32,.36,.4,.44,.48,.52,.56,.60,.64,.68,.72]
        ytick_strs = ['0.20','0.24','0.28','0.32','0.36','0.40','0.44','0.48','0.52','0.56','0.60','0.64','0.68','0.72']
        ylabel = 'AUC'
        sublabel_text_x = .05
        sublabel_text_y = .685
        sn_text = .69


    if plot_metric=='day_csi':
        ylims = [.1,.5]
        ytick_nums = [.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6]
        ytick_strs = ['0.10','0.15','0.20','0.25','0.30','0.35','0.40','0.45','0.50','0.55','0.60']
        ylabel = 'Max CSI'
        sublabel_text_x = .02
        sublabel_text_y = .55
        sn_text = .55

    
    matplotlib.rcParams['axes.facecolor'] = [0.95,0.95,0.95] 
    fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(20,20))
    axes[0].grid(axis='y',zorder=0)
    axes[0].set_axisbelow(True)
    axes[1].grid(axis='y',zorder=0)
    axes[1].set_axisbelow(True)
    xticks = np.arange(24)
    axes[0].set_xticks(xticks,['','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','' ],fontsize=18,rotation=45)
    axes[0].tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=True)
    axes[1].set_xticks(xticks,['','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','',
                               '','Day 1','Day 2','Day 3','Day 4','' ],fontsize=18,rotation=45)
    axes[1].tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=True)
    axes[0].set_ylabel(ylabel,fontsize=24)
    axes[0].set_ylim(ylims)
    axes[0].set_yticks(ytick_nums,ytick_strs,fontsize=18)
    axes[1].set_ylabel(ylabel,fontsize=24)
    axes[1].set_ylim(ylims)
    axes[1].set_yticks(ytick_nums,ytick_strs,fontsize=18)
    axes[0].text(1.5,sn_text,'Summer',fontsize=24)
    axes[1].text(1.5,sn_text,'Summer',fontsize=24)
    axes[0].text(7.5,sn_text,'Fall',fontsize=24)
    axes[1].text(7.5,sn_text,'Fall',fontsize=24)
    axes[0].text(13.5,sn_text,'Winter',fontsize=24)
    axes[1].text(13.5,sn_text,'Winter',fontsize=24)
    axes[0].text(19.5,sn_text,'Spring',fontsize=24)
    axes[1].text(19.5,sn_text,'Spring',fontsize=24)

    colors_hex = ['#ca0020','#f4a582','#92c5de','#0571b0']
    width = 1
    position=1
    for season in seasons:
        unet_ssn_stats = unet_stats[season]
        for key in unet_ssn_stats['Day 1']:
            print(key)
            print(type(unet_ssn_stats['Day 1'][key]))
        lstm_ssn_stats = lstm_stats[season]
        for day in range(1,5):
            unet_day_stat = np.max(unet_ssn_stats['Day %s'%day][plot_metric])#.numpy()
            lstm_day_stat = np.max(lstm_ssn_stats['Day %s'%day][plot_metric])#.numpy()
            axes[0].bar(x=position,height=unet_day_stat,width=width,edgecolor='black',color=colors_hex[day-1],alpha=1,zorder=3)
            axes[1].bar(x=position,height=lstm_day_stat,width=width,edgecolor='black',color=colors_hex[day-1],alpha=1,zorder=3)
            axes[0].text(position-.4,unet_day_stat+.01,f"{unet_day_stat:.2f}",fontsize=18)
            axes[1].text(position-.4,lstm_day_stat+.01,f"{lstm_day_stat:.2f}",fontsize=18)
            position+=1
        position+=2
    axes[0].text(sublabel_text_x,sublabel_text_y,'(a)',fontsize = 24,fontweight = 'bold')
    axes[1].text(sublabel_text_x,sublabel_text_y,'(b)',fontsize = 24,fontweight = 'bold')
    axes[0].set_title('U-Net',fontsize=24)
    axes[1].set_title('LSTM',fontsize=24)
    plt.savefig('./AIES_results_reviewer_edits_v2/seasonal_eval_%s_11Jan26.png'%plot_metric)
    plt.savefig('./AIES_results_reviewer_edits_v2/seasonal_eval_%s_11Jan26.pdf'%plot_metric)
    plt.close()

def main():
    tf.config.set_visible_devices([], 'GPU')
    print('NO VISIBLE DEVICES!!!!')
    generate_plot2(plot_metric='day_auc')
    generate_plot2(plot_metric='day_csi')

if __name__=='__main__':
    main()
    print('ENDED SUCCESSFULLY')