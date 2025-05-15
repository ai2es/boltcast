import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys
import pickle
import glob
import pandas as pd

def main():
    print('6 BC_plot_perm_importance.py')
    print('the modules loaded correctly')
    
    metric = 'auc_PR'#'auc_ROC','binary_accuracy','precision','recall']
    threshold=.25
    print('Generating the AIES figure')
    AIES_perm_imp(metric=metric,
                    threshold=.25,
                    lstm_deep=1)
    

def get_boxplot_values(feature='cape',
                        metric='binary_accuracy',
                        threshold=.25,
                        rotation=4,
                        model='UNet',
                        lstm_deep=0):

    print('get boxplot values',feature, metric, threshold, rotation, model)
    load_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/model_analysis_output_9Jan25/single_perm_results_pkl/'
    if model =='UNet':
        glob_files = '%s_perm_num_*_%s_rot_%s_threshold_%s.pkl'%(feature,model,rotation,threshold)
    if model == 'LSTM':
        glob_files = '%s_perm_num_*_%s_rot_%s_lstm_deep_%s_threshold_%s.pkl'%(feature,model,rotation,lstm_deep,threshold)
    glob_call = load_dir+glob_files
    files = glob.glob(glob_call)
    perm_list = []
    non_perm_list = []
    for f,file in enumerate(files):
        perm_dict = pickle.load(open(file,'rb'))
        perm_list.append(perm_dict[metric])
        non_perm_list.append(perm_dict['No_Perm'][metric])
    metric_diff = np.array(non_perm_list)-np.array(perm_list)
    return metric_diff

def plot_single_whisker(feature='cape',
                        metric='binary_accuracy',
                        threshold=.25,
                        rotation=4,
                        model='UNet',
                        lstm_deep=0):
    
    single_data = get_boxplot_values(feature=feature,
                                    metric=metric,
                                    threshold=threshold,
                                    rotation=rotation,
                                    model=model,
                                    lstm_deep=lstm_deep)

    fig,ax = plt.subplots(nrows=1,
                        ncols=1,
                        figsize=(10,8))
    bplot = plt.boxplot(single_data,
                        vert=False,
                        patch_artist=True,
                        medianprops=dict(color='black'))
    plt.ylabel('%s Difference'%(metric.upper()),fontsize=18)
    ax.set_xticks(ticks=[])

    xticks = np.linspace(0,.1,11)
    x_tick_labels = []
    for xtick in xticks:
        x_tick_labels.append(f"{xtick:02}")
    ax.set_xticks(ticks=xticks,labels=x_tick_labels,fontsize=18)
    plt.grid('on')
    plt.title(feature.upper(),fontsize=24)

    colors_rgb = [(247,244,249),(231,225,239),(212,185,218),(201,148,199),(223,101,176),(231,41,138),(206,18,86),(152,0,67),(103,0,31)]
    colors_hex = ['#f7f4f9','#e7e1ef','#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f']
    
    for patch, color in zip(bplot['boxes'], colors_hex):
        patch.set_facecolor(color)
    save_dir = './single_perm_results_one_boxplot/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    if model=='UNet':
        fsave = '%s_rot_%s_%s_thresh_%s_%s.png'%(feature,rotation,model,threshold,metric)
    else:
        fsave = '%s_rot_%s_%s_thresh_%s_lstm_deep_%s_%s.png'%(feature,rotation,model,threshold,lstm_deep,metric)
    plt.savefig(save_dir+fsave)
    plt.close()

def plot_all_whisker(metric='binary_accuracy',
                        threshold=.25,
                        rotation=4,
                        model='UNet',
                        lstm_deep=0):

    matplotlib.rcParams['axes.facecolor'] = [0.8,0.8,0.8] #makes a grey background to the axis face
    matplotlib.rcParams['axes.titlesize'] = 18 
    matplotlib.rcParams['xtick.labelsize'] = 18 
    matplotlib.rcParams['ytick.labelsize'] = 18 
    matplotlib.rcParams['legend.fontsize'] = 18 

    print('plot_all_whisker')
    features = ['cape','reflectivity','precip_rate','lifted_idx','w','rain_q','ice_q','snow_q','graupel_q']
    box_stack_np = np.zeros([25,9])
    
    for f,feature in enumerate(features):
        box_stack_np[:,f] = get_boxplot_values(feature=feature,
                                    metric=metric,
                                    threshold=threshold,
                                    rotation=rotation,
                                    model=model,
                                    lstm_deep=lstm_deep)
    
    box_stack_df = pd.DataFrame(box_stack_np,columns=features)
    meds = box_stack_df.median(axis=0)
    meds = meds.sort_values(ascending=True)
    box_stack_df = box_stack_df[meds.index]#sorted by median

    labels_features = box_stack_df.columns
    data = box_stack_df.values
    
    fig,ax = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=(12,8))
    
    bplot = ax.boxplot(data,
                        vert=False,
                        patch_artist=True,
                        tick_labels=labels_features,
                        medianprops=dict(color='black'))
    box_yticks = ax.get_yticks()
    colors_hex = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d']
    for patch, color in zip(bplot['boxes'], colors_hex):
        patch.set_facecolor(color)
    plt.barh(box_yticks,meds,color=colors_hex,alpha=.5)
    plt.grid('on')
    plt.vlines(0,ymin=0,ymax=10,linewidth=1,color='black')
    plt.ylim([0,10])
    if metric=='auc_ROC':
        plt.xlim([-.005,.3])
        xlabel = 'AUC_ROC'
    elif metric=='auc_PR':
        plt.xlim([-.05,.3])
        xlabel = 'AUC_PR'
    elif metric=='binary_accuracy':
        plt.xlim([-.01,.05])
        xlabel='Binary Accuracy'
    elif metric=='recall':
        plt.xlim([-.05,.35])
        xlabel='Recall'
    else:
        plt.xlim([-.025,.125])
        xlabel='Precision'
    plt.xlabel('$\Delta$ %s'%(xlabel),fontsize=18,fontweight='bold')
    plt.ylabel('Model Inputs',fontsize=18,fontweight='bold')
    plt.title('Permutation Importance (n=25)',fontsize=24)
    plt.tight_layout()
    save_dir_png = './single_perm_results_png/'
    save_dir_pdf = './single_perm_results_pdf/'
    if os.path.isdir(save_dir_png)==False:
        os.makedirs(save_dir_png)
    if os.path.isdir(save_dir_pdf)==False:
        os.makedirs(save_dir_pdf)
    if model=='UNet': 
        fsave_png = 'all_features_perm_whisker_%s_rot_%s_%s_threshold_%s.png'%(metric,rotation,model,threshold)
        fsave_pdf = 'all_features_perm_whisker_%s_rot_%s_%s_threshold_%s.pdf'%(metric,rotation,model,threshold)
    if model=='LSTM':
        fsave_png = 'all_features_perm_whisker_%s_rot_%s_%s_lstm_deep_%s_threshold_%s.png'%(metric,rotation,model,lstm_deep,threshold)
        fsave_pdf = 'all_features_perm_whisker_%s_rot_%s_%s_lstm_deep_%s_threshold_%s.pdf'%(metric,rotation,model,lstm_deep,threshold)
    plt.savefig(save_dir_png+fsave_png)
    plt.savefig(save_dir_pdf+fsave_pdf)
    plt.close()

def plot_all_bar_whisker_dual_model(metric='binary_accuracy',
                        threshold=.25,
                        rotation=4,
                        lstm_deep=1):

    matplotlib.rcParams['axes.facecolor'] = [0.8,0.8,0.8] #makes a grey background to the axis face
    matplotlib.rcParams['axes.titlesize'] = 18 
    matplotlib.rcParams['xtick.labelsize'] = 18 
    matplotlib.rcParams['ytick.labelsize'] = 18 
    matplotlib.rcParams['legend.fontsize'] = 18 

    print('plot_all_whisker_both_models')
    features = ['cape','reflectivity','precip_rate','lifted_idx','w','rain_q','ice_q','snow_q','graupel_q']
    box_stack_np_unet = np.zeros([25,9])
    box_stack_np_lstm = np.zeros([25,9])
    for f,feature in enumerate(features):
        box_stack_np_unet[:,f] = get_boxplot_values(feature=feature,
                                    metric=metric,
                                    threshold=threshold,
                                    rotation=rotation,
                                    model='UNet',
                                    lstm_deep=0)
        
        box_stack_np_lstm[:,f] = get_boxplot_values(feature=feature,
                                    metric=metric,
                                    threshold=threshold,
                                    rotation=rotation,
                                    model='LSTM',
                                    lstm_deep=lstm_deep)
    
    box_stack_df_unet = pd.DataFrame(box_stack_np_unet,columns=features)
    meds_unet = box_stack_df_unet.median(axis=0)
    meds_unet = meds_unet.sort_values(ascending=True)
    box_stack_df_unet = box_stack_df_unet[meds_unet.index]#sorted by median
    labels_features_unet = box_stack_df_unet.columns
    data_unet = box_stack_df_unet.values

    box_stack_df_lstm = pd.DataFrame(box_stack_np_lstm,columns=features)
    meds_lstm = box_stack_df_lstm.median(axis=0)
    # meds_unet = meds_unet.sort_values(ascending=True)
    # box_stack_df_unet = box_stack_df_unet[meds_unet.index]#sorted by median
    data_lstm = box_stack_df_lstm.values
    meds_lstm_2unet = []
    data_lstm_sorted = []
    for i,feature in enumerate(labels_features_unet):
        meds_lstm_2unet.append(meds_lstm[feature])
        data_lstm_sorted.append(box_stack_df_lstm[feature].values)
    fig,ax = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=(12,8))
    hgt = .4
    unet_positions = np.arange(9)
    lstm_positions = unet_positions-hgt
    
    colors_hex_unet = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d']
    colors_hex_lstm = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b']
    bplot_unet = ax.boxplot(data_unet,
                        positions=unet_positions,
                        vert=False,
                        patch_artist=True,
                        tick_labels=labels_features_unet,
                        widths = .3,
                        medianprops=dict(color='black'))
    for patch, color in zip(bplot_unet['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
    bplot_lstm = ax.boxplot(data_lstm_sorted,
                            positions=lstm_positions,
                            vert=False,
                            patch_artist=True,
                            widths=.3,
                            tick_labels=['','','','','','','','',''],
                            manage_ticks=False,
                            medianprops=dict(color='black'))
    for patch,color in zip(bplot_lstm['boxes'],colors_hex_lstm):
        patch.set_facecolor(color)
    plt.grid('on',axis='x')
    unet_bar_handles = ax.barh(y=unet_positions,
            height=hgt,
            color=colors_hex_unet[6],
            alpha=.8,
            hatch='\\',
            label='UNet',
            width=meds_unet)
    lstm_bar_handles = ax.barh(y=lstm_positions,
            height=hgt,
            color=colors_hex_lstm[6],
            alpha=.8,
            label='LSTM',
            hatch='/',
            width=meds_lstm_2unet)
    plt.vlines(0,ymin=-1,ymax=8.5,linewidth=2,color='black')
    plt.ylim([-1,8.5])
    if metric=='auc_ROC':
        plt.xlim([-.005,.3])
        xlabel = 'AUC_ROC'
    elif metric=='auc_PR':
        plt.xlim([-.05,.3])
        xlabel = 'AUC_PR'
    elif metric=='binary_accuracy':
        plt.xlim([-.01,.05])
        xlabel='Binary Accuracy'
    elif metric=='recall':
        plt.xlim([-.05,.35])
        xlabel='Recall'
    else:
        plt.xlim([-.025,.125])
        xlabel='Precision'
    plt.legend([unet_bar_handles[8],lstm_bar_handles[8]],['UNet','LSTM'])
    plt.xlabel('$\Delta$ %s'%(xlabel),fontsize=18,fontweight='bold')
    plt.ylabel('Model Inputs',fontsize=18,fontweight='bold')
    plt.title('Permutation Importance (n=25), Threshold: %s'%(threshold),fontsize=24)
    plt.tight_layout()
    save_dir_png = './dual_model_single_perm_results_png/'
    fsave_png = 'dual_bar_perm_imp_rot_%s_threshold_%s.png'%(rotation,threshold)

    save_dir_pdf = './dual_model_single_perm_results_pdf/'
    fsave_pdf = 'dual_bar_perm_imp_rot_%s_threshold_%s.pdf'%(rotation,threshold)

    if os.path.isdir(save_dir_png)==False:
        os.makedirs(save_dir_png)
    if os.path.isdir(save_dir_pdf)==False:
        os.makedirs(save_dir_pdf)
    plt.savefig(save_dir_png+fsave_png)
    plt.savefig(save_dir_pdf+fsave_pdf)
    plt.close()

def AIES_perm_imp(metric='binary_accuracy',
                    threshold=.25,
                    lstm_deep=1):

    print('AIES_perm_imp')

    #set the plotting parameters
    matplotlib.rcParams['axes.facecolor'] = [0.8,0.8,0.8] #makes a grey background to the axis face
    matplotlib.rcParams['axes.titlesize'] = 18 
    matplotlib.rcParams['xtick.labelsize'] = 18 
    matplotlib.rcParams['ytick.labelsize'] = 18 
    matplotlib.rcParams['legend.fontsize'] = 18 

    #set the common variables for the analysis
    features = ['cape','reflectivity','precip_rate','lifted_idx','w','rain_q','ice_q','snow_q','graupel_q']
    rotations = [0,1,2,3,4]
    box_stack_np_unet = np.zeros([5,25,9])
    box_stack_np_lstm = np.zeros([5,25,9])

    #get the metric differences for each rotation and feature
    for f,feature in enumerate(features):
        for r,rotation in enumerate(rotations):

            #store the unet metric differences
            box_stack_np_unet[r,:,f] = get_boxplot_values(feature=feature,
                                        metric=metric,
                                        threshold=threshold,
                                        rotation=rotation,
                                        model='UNet',
                                        lstm_deep=0)
            
            #store the lstm metric differences
            box_stack_np_lstm[r,:,f] = get_boxplot_values(feature=feature,
                                        metric=metric,
                                        threshold=threshold,
                                        rotation=rotation,
                                        model='LSTM',
                                        lstm_deep=lstm_deep)

    #store the data per feature for all rotations
    box_stack_unet = np.zeros([125,9])
    box_stack_lstm = np.zeros([125,9])
    for f,feature in enumerate(features):
        box_stack_unet[:,f] = np.ravel(box_stack_np_unet[:,:,f])
        box_stack_lstm[:,f] = np.ravel(box_stack_np_lstm[:,:,f])
    box_stack_np_unet = box_stack_unet
    box_stack_np_lstm = box_stack_lstm

    #create a dataframe for easier statistic counting
    #calculate the median metric difference values for the unet
    #then sort the dataframe by the median values
    box_stack_df_unet = pd.DataFrame(box_stack_np_unet,columns=features)
    meds_unet = box_stack_df_unet.median(axis=0)
    meds_unet = meds_unet.sort_values(ascending=True)
    box_stack_df_unet = box_stack_df_unet[meds_unet.index]#sorted by median
    labels_features_unet = box_stack_df_unet.columns
    print(labels_features_unet)
    data_unet = box_stack_df_unet.values

    box_stack_df_lstm = pd.DataFrame(box_stack_np_lstm,columns=features)
    meds_lstm = box_stack_df_lstm.median(axis=0)
    # meds_unet = meds_unet.sort_values(ascending=True)
    # box_stack_df_unet = box_stack_df_unet[meds_unet.index]#sorted by median
    data_lstm = box_stack_df_lstm.values
    meds_lstm_2unet = []
    data_lstm_sorted = []
    for i,feature in enumerate(labels_features_unet):
        meds_lstm_2unet.append(meds_lstm[feature])
        data_lstm_sorted.append(box_stack_df_lstm[feature].values)

    # create the plot
    fig,ax = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=(12,8))
    hgt = .4

    unet_positions = np.arange(9)
    lstm_positions = unet_positions-hgt
    
    colors_hex_unet = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d']
    colors_hex_lstm = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b']
    
    ##hard coded for clean formatting of the final figure. i know the sorted labels. 
    y_labels = np.flip(['CAPE','Reflectivity','Precipitation Rate','Lifted Index','Snow Mixing Ratio','Vertical Velocity','Rain Mixing Ratio','Ice Mixing Ratio','Graupel Mixing Ratio'])
    
    bplot_unet = ax.boxplot(data_unet,
                        positions=unet_positions,
                        vert=False,
                        patch_artist=True,
                        tick_labels=y_labels,
                        widths = .3,
                        medianprops=dict(color='black'))
    for patch, color in zip(bplot_unet['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
    bplot_lstm = ax.boxplot(data_lstm_sorted,
                            positions=lstm_positions,
                            vert=False,
                            patch_artist=True,
                            widths=.3,
                            tick_labels=['','','','','','','','',''],
                            manage_ticks=False,
                            medianprops=dict(color='black'))
    for patch,color in zip(bplot_lstm['boxes'],colors_hex_lstm):
        patch.set_facecolor(color)
    plt.grid('on',axis='x')
    unet_bar_handles = ax.barh(y=unet_positions,
            height=hgt,
            color=colors_hex_unet,
            alpha=.8,
            hatch=None,
            label='UNet',
            width=meds_unet)
    lstm_bar_handles = ax.barh(y=lstm_positions,
            height=hgt,
            color=colors_hex_lstm,
            alpha=.8,
            label='LSTM',
            hatch=None,
            width=meds_lstm_2unet)
    plt.vlines(0,ymin=-1,ymax=8.5,linewidth=2,color='black')


    plt.ylim([-1,8.5])
    if metric=='auc_ROC':
        plt.xlim([-.005,.3])
        xlabel = 'AUC_ROC'
    elif metric=='auc_PR':
        plt.xlim([-.05,.5])
        xlabel = 'AUC'
    elif metric=='binary_accuracy':
        plt.xlim([-.01,.05])
        xlabel='Binary Accuracy'
    elif metric=='recall':
        plt.xlim([-.05,.35])
        xlabel='Recall'
    else:
        plt.xlim([-.025,.125])
        xlabel='Precision'
    plt.legend([unet_bar_handles[8],lstm_bar_handles[8]],['UNet','LSTM'])
    plt.xlabel('$\Delta$ %s'%(xlabel),fontsize=18,fontweight='bold')
    plt.ylabel('Model Inputs',fontsize=18,fontweight='bold')
    plt.title('Permutation Importance (n=25), Threshold: %s'%(threshold),fontsize=24)
    plt.tight_layout()

    save_dir_png = './dual_model_single_perm_results_png/'
    fsave_png = 'dual_bar_perm_imp_threshold_%s.png'%(threshold)
    save_dir_pdf = './dual_model_single_perm_results_pdf/'
    fsave_pdf = 'dual_bar_perm_imp_threshold_%s.pdf'%(threshold)

    if os.path.isdir(save_dir_png)==False:
        os.makedirs(save_dir_png)
    if os.path.isdir(save_dir_pdf)==False:
        os.makedirs(save_dir_pdf)
    plt.savefig(save_dir_png+fsave_png)
    plt.savefig(save_dir_pdf+fsave_pdf)
    plt.close()

if __name__=='__main__':

    # visible_devices = tf.config.get_visible_devices('GPU') 
    # n_visible_devices = len(visible_devices)
    # print(n_visible_devices)
    # tf.config.set_visible_devices([], 'GPU')
    # print('GPU turned off')

    main()
