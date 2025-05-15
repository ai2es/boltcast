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
                    threshold=threshold,
                    lstm_deep=1)
    

def get_boxplot_values(feature='cape',
                        rotation=4,
                        model='UNet'):

    print('get boxplot values',feature, rotation, model)
    load_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/BoltCast_permutations/'
    if model=='UNet':
        glob_files = '%s_rot_%s_%s_perm_num_100_exp_*.pkl'%(feature,rotation,model)
    if model=='LSTM':
        glob_files = '%s_rot_%s_%s_perm_num_100_conv_deep_0_lstm_deep_1_exp_*.pkl'%(feature,rotation,model)
    print(glob_files)
    glob_call = load_dir+glob_files
    files = glob.glob(glob_call)
    print(files)
    perm_list = []
    non_perm_list = []
    for fi,file in enumerate(files):
        if fi==0:
            perm_df = pickle.load(open(file,'rb'))
            non_perm_np = perm_df['auc_PR'].iloc[0]
            perm_metrics = perm_df['auc_PR'].iloc[1:101]
        else:
            perm_metrics = pd.concat([perm_metrics,perm_df['auc_PR'].iloc[1:101]])
    print(perm_metrics)
    metric_diff = non_perm_np - perm_metrics.values
    print(metric_diff)
    return metric_diff

def AIES_perm_imp(metric='auc_PR',
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
    rotation = 4
    box_stack_np_unet = np.zeros([5,1000,9])#permutations,features,rotations
    box_stack_np_lstm = np.zeros([5,1000,9])

    #get the metric differences for each rotation and feature
    for r,rotation in enumerate([0,1,2,3,4]):
        for f,feature in enumerate(features):
            if f>=0:

                #store the unet metric differences
                box_stack_np_unet[r,:,f] = get_boxplot_values(feature=feature,
                                            rotation=rotation,
                                            model='UNet')
                
                #store the lstm metric differences
                box_stack_np_lstm[r,:,f] = get_boxplot_values(feature=feature,
                                            rotation=rotation,
                                            model='LSTM')

    box_stack_unet_2 = np.zeros([box_stack_np_unet.shape[0]*box_stack_np_unet.shape[1],9])
    box_stack_lstm_2 = np.zeros([box_stack_np_lstm.shape[0]*box_stack_np_lstm.shape[1],9])

    for f,feature in enumerate(features):
        box_stack_unet_2[:,f] = np.ravel(box_stack_np_unet[:,:,f])
        box_stack_lstm_2[:,f] = np.ravel(box_stack_np_lstm[:,:,f])
    del box_stack_np_lstm, box_stack_np_unet

    box_stack_np_unet = box_stack_unet_2
    box_stack_np_lstm = box_stack_lstm_2

    #create a dataframe for easier statistic counting
    #calculate the median metric difference values for the unet
    #then sort the dataframe by the median values
    box_stack_df_unet = pd.DataFrame(box_stack_np_unet,columns=features)
    meds_unet = box_stack_df_unet.median(axis=0)
    meds_unet = meds_unet.sort_values(ascending=True)
    print('sorted_unet_medians')
    print('meds_unet:',meds_unet)
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

    # create the plot
    fig,ax = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=(12,8))
    hgt = .4

    unet_positions = np.arange(9)
    lstm_positions = unet_positions-hgt
    
    colors_hex_unet = ['#ca0020','#ca0020','#ca0020','#ca0020','#ca0020','#ca0020','#ca0020','#ca0020','#ca0020']
    colors_hex_lstm = ['#0571b0','#0571b0','#0571b0','#0571b0','#0571b0','#0571b0','#0571b0','#0571b0','#0571b0']
    
    ##hard coded for clean formatting of the final figure. I know the sorted labels. 
    y_labels = np.flip(['CAPE','Reflectivity','Precipitation Rate','Lifted Index','Snow Mixing Ratio','Vertical Velocity','Ice Mixing Ratio','Rain Mixing Ratio','Graupel Mixing Ratio'])
    bplot_unet = ax.boxplot(data_unet,
                        positions=unet_positions,
                        vert=False,
                        patch_artist=True,
                        tick_labels=y_labels,
                        widths = .3,
                        medianprops=dict(color='black'))

    for patch, color in zip(bplot_unet['boxes'], colors_hex_unet):
        print('unet_box_color',color)
        patch.set_facecolor(color)
        patch.set_hatch('//')
    bplot_lstm = ax.boxplot(data_lstm_sorted,
                            positions=lstm_positions,
                            vert=False,
                            patch_artist=True,
                            widths=.3,
                            tick_labels=['','','','','','','','',''],
                            manage_ticks=False,
                            medianprops=dict(color='black'))
    for patch,color in zip(bplot_lstm['boxes'],colors_hex_lstm):
        print('lstm_box_color:',color)
        patch.set_facecolor(color)
        patch.set_hatch('\\')

    plt.grid('on',axis='x')
    unet_bar_handles = ax.barh(y=unet_positions,
            height=hgt,
            color=colors_hex_unet,
            alpha=.4,
            hatch=None,
            label='UNet',
            width=meds_unet)
    lstm_bar_handles = ax.barh(y=lstm_positions,
            height=hgt,
            color=colors_hex_lstm,
            alpha=.4,
            label='LSTM',
            hatch=None,
            width=meds_lstm_2unet)
    plt.vlines(0,ymin=-1,ymax=8.5,linewidth=2,color='black')

    plt.ylim([-1,8.5])
    plt.xlim([-.05,.5])
    xlabel = 'AUC'

    plt.legend([unet_bar_handles[8],lstm_bar_handles[8]],['UNet','LSTM'])
    plt.xlabel('$\Delta$ %s'%(xlabel),fontsize=18,fontweight='bold')
    plt.ylabel('Model Inputs',fontsize=18,fontweight='bold')
    plt.title('Permutation Importance (n=1000)',fontsize=24)
    plt.tight_layout()

    save_dir = './dual_model_single_perm_results/'
    fsave_png = 'dual_bar_perm_imp_threshold_%s_conv_deep_0_lstm_deep_1_no_bar.png'%(threshold)
    fsave_pdf = 'dual_bar_perm_imp_threshold_%s_conv_deep_0_lstm_deep_1_no_bar.pdf'%(threshold)

    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    plt.savefig(save_dir+fsave_png)
    plt.savefig(save_dir+fsave_pdf)
    plt.close()

if __name__=='__main__':
    main()
