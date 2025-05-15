import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import shutil
import glob

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
        ytick_nums = [.2,.22,.24,.26,.28,.30,.32,.34,.36,.38,.4,.42,.44,.46,.48,.50,.52,.54,.56,.58,.60,.62,.64,.66,.68,.70]
        ytick_strs = ['0.20','0.22','0.24','0.26','0.28','0.30','0.32','0.34','0.36','0.38','0.40','0.42','0.44','0.46','0.48','0.50','0.52','0.54','0.56','0.58','0.60','0.62','0.64','0.66','0.68','0.70']
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
    bplot = axes[0].boxplot(unet_csis[0,:,:], positions=[1,2,3,4],vert=True, patch_artist=True, widths=1, medianprops=dict(color='black'))
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
    bplot = axes[0].boxplot(unet_csis[1,:,:], positions=[7,8,9,10],vert=True, patch_artist=True, widths=1, medianprops=dict(color='black'))
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
    bplot = axes[0].boxplot(unet_csis[2,:,:], positions=[13,14,15,16],vert=True, patch_artist=True, widths=1, medianprops=dict(color='black'))
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
    bplot = axes[0].boxplot(unet_csis[3,:,:], positions=[19,20,21,22],vert=True, patch_artist=True, widths=1, medianprops=dict(color='black'))
    for patch, color in zip(bplot['boxes'], colors_hex_unet):
        patch.set_facecolor(color)
        patch.set_hatch('//')
    del bplot


    axes[0].set_ylabel(ylabel,fontsize=24)
    axes[0].set_ylim(ylims)
    axes[0].set_yticks(ytick_nums,ytick_strs,fontsize=18)
    axes[0].set_title('UNet',fontsize=24)
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
    bplot = axes[1].boxplot(lstm_csis[0,:,:], positions=[1,2,3,4],vert=True, patch_artist=True, widths=1, medianprops=dict(color='black'))
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
    bplot = axes[1].boxplot(lstm_csis[1,:,:], positions=[7,8,9,10],vert=True, patch_artist=True, widths=1, medianprops=dict(color='black'))
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
    bplot = axes[1].boxplot(lstm_csis[2,:,:], positions=[13,14,15,16],vert=True, patch_artist=True, widths=1, medianprops=dict(color='black'))
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
    bplot = axes[1].boxplot(lstm_csis[3,:,:], positions=[19,20,21,22],vert=True, patch_artist=True, widths=1, medianprops=dict(color='black'))
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
    plt.savefig('./seasonal_csis/seasonal_csi_lead_time_nobar_%s.pdf'%(plot_metric))
    plt.savefig('./seasonal_csis/seasonal_csi_lead_time_nobar_%s.png'%(plot_metric))
    plt.close()

def main():
    print('generating csi figure')
    generate_plot(plot_metric='day_csi')
    generate_plot(plot_metric='day_auc')

if __name__=='__main__':
    main()