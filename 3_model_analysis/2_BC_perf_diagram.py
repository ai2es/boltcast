import sys
import argparse
import pickle
import pandas as pd
import wandb
import socket
import matplotlib.pyplot as plt
import shutil 
import tensorflow as tf
from sklearn.metrics import auc, precision_recall_curve
import os
from gewitter_functions import *
import xarray as xr

#load contingency_table func
from gewitter_functions import get_contingency_table,make_performance_diagram_axis,get_acc,get_pod,get_sr,csi_from_sr_and_pod
import matplotlib
import matplotlib.patheffects as path_effects

#outlines for text 
pe1 = [path_effects.withStroke(linewidth=1.5,
                            foreground="k")]
pe2 = [path_effects.withStroke(linewidth=1.5,
                            foreground="w")]

matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9] #makes a grey background to the axis face
matplotlib.rcParams['axes.labelsize'] = 24 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 24 
matplotlib.rcParams['xtick.labelsize'] = 18 
matplotlib.rcParams['ytick.labelsize'] = 18 
matplotlib.rcParams['legend.fontsize'] = 18 
matplotlib.rcParams['legend.facecolor'] = '#f7f7f7'#light grey
matplotlib.rcParams['savefig.transparent'] = False

def overall_gewitter_LSTM(lstm_deep=1):
    print('building the gewitter plots for overall LSTM performance')

    thresh = np.arange(0.05,1.05,0.05)
    rotations = [0,1,2,3,4]
    conv_deeps = [0,1,2]
    
    for conv_deep in conv_deeps:
        #plot it up  
        fig, ax = plt.subplots(1,1,figsize=(20,20))
        ax = make_performance_diagram_axis(ax, csi_cmap='Greys_r')
        colors = ['#fee5d9','#fcae91','#fb6a4a','#de2d26','#a50f15']

        # colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442']  # Colorblind-friendly
        linestyles = ['-', '--', '-.', ':']
        markers = ['s', 'o', '^', 'D']

        for r,rot in enumerate(rotations):
            dir_out = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/labels_outputs/'
            file_load = 'BC_LSTM_rot_%s_conv_4_conv_deep_%s_lstm_deep_%s_no_drop_no_shuffle_model_output_labels.pkl'%(rot,conv_deep,lstm_deep)
            rot_dict = pickle.load(open(dir_out+file_load,'rb'))
            model_output = rot_dict['model_output']
            labels = rot_dict['labels']
            del rot_dict

            outputs_1d = model_output.ravel()
            labels_1d = labels.ravel()
            del model_output, labels
            
            #statistics we need for performance diagram 
            tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())
            fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())
            fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())

            # get performance diagram line by getting tp,fp and fn 
            tp.reset_state()
            fp.reset_state()
            fn.reset_state()

            tps = tp(labels_1d,outputs_1d)
            fps = fp(labels_1d,outputs_1d)
            fns = fn(labels_1d,outputs_1d)

            # calc x,y of performance diagram 
            pods = tps/(tps + fns)
            srs = tps/(tps + fps)
            csis = tps/(tps + fns + fps)

            label = 'Max CSI (rot %s): %s'%(rot,f"{max(csis):.2f}")
            ax.plot(np.asarray(srs),np.asarray(pods),'-s',
                    color=colors[r],
                    markerfacecolor=colors[r],
                    label=label)

            # for i,t in enumerate(thresh):
            #     if i==5:
            #         text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
            #         ax.text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
            #     if i==10:
            #         text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
            #         ax.text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
            #     if i==15:
            #         text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
            #         ax.text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
        
        print('saving gewitter plot in overall_gewitter_UNet()')
        ax.legend()
        plt.tight_layout()
        plt.savefig('./overall_performance_diagrams/LSTM_deep_%s_conv_deep_%s_all_rot.png'%(lstm_deep,conv_deep))
        plt.savefig('./overall_performance_diagrams/LSTM_deep_%s_conv_deep_%s_all_rot.pdf'%(lstm_deep,conv_deep))
        plt.close()

def overall_gewitter_UNet():
    print('building the gewitter plots for overall UNet performance')
    dir_out = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/UNet_symmetric/all_rots/labels_outputs/'
    rotations = [0,1,2,3,4]
    

    #plot it up  
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    ax = make_performance_diagram_axis(ax, csi_cmap='Blues_r')
    colors = ['#fee5d9','#fcae91','#fb6a4a','#de2d26','#a50f15']
    thresh = thresh = np.arange(0.05,1.05,0.05)

    for r,rot in enumerate(rotations):
        file_load = 'UNet_symmetric_rot_%s_output.pkl'%(rot)
        rot_dict = pickle.load(open(dir_out+file_load,'rb'))
        model_output = rot_dict['model_output']
        labels = rot_dict['labels']
        del rot_dict

        outputs_1d = model_output.ravel()
        labels_1d = labels.ravel()
        del model_output, labels
        
        #statistics we need for performance diagram 
        tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())
        fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())
        fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())

        # get performance diagram line by getting tp,fp and fn 
        tp.reset_state()
        fp.reset_state()
        fn.reset_state()

        tps = tp(labels_1d,outputs_1d)
        fps = fp(labels_1d,outputs_1d)
        fns = fn(labels_1d,outputs_1d)

        # calc x,y of performance diagram 
        pods = tps/(tps + fns)
        srs = tps/(tps + fps)
        csis = tps/(tps + fns + fps)

        label = 'max_CSI_%s_rot_%s_UNet'%(f"{max(csis):.2f}",rot)
        ax.plot(np.asarray(srs),np.asarray(pods),'-s',
                color=colors[r],
                markerfacecolor=colors[r],
                label=label)

        # for i,t in enumerate(thresh):
        #     if i==5:
        #         text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        #         ax.text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
        #     if i==10:
        #         text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        #         ax.text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
        #     if i==15:
        #         text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        #         ax.text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
    
    print('saving gewitter plot in overall_gewitter_UNet()')
    ax.legend()
    plt.tight_layout()
    plt.savefig('./performance_diagrams/UNet_symmetric_all_rot.png')
    plt.savefig('./performance_diagrams/UNet_symmetric_all_rot.pdf')
    plt.close()

def daily_gewitter_UNet(rotation=4):

    # visible_devices = tf.config.get_visible_devices('GPU') 
    # n_visible_devices = len(visible_devices)
    # print(n_visible_devices)
    # tf.config.set_visible_devices([], 'GPU')
    # print('GPU turned off')

    unet_out_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/UNet/labels_outputs/'
    file_load = 'UNet_symmetric_rot_%s_output.pkl'%(rotation)

    rot_dict = pickle.load(open(unet_out_dir+file_load,'rb'))
    model_output = rot_dict['model_output']
    labels = rot_dict['labels']

    day_1_output = model_output[:,0,:,:]
    day_1_labels = labels[:,0,:,:]

    day_2_output = model_output[:,1,:,:]
    day_2_labels = labels[:,1,:,:]

    day_3_output = model_output[:,2,:,:]
    day_3_labels = labels[:,2,:,:]

    day_4_output = model_output[:,3,:,:]
    day_4_labels = labels[:,3,:,:]

    #plot it up  
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    ax = make_performance_diagram_axis(ax)
    colors = ['#7b3294','#c2a5cf','#a6dba0','#008837'] 
    # ['#fee5d9','#fcae91','#fb6a4a','#cb181d']#reds
    thresh = np.arange(0.05,1.05,0.05)

    #statistics we need for performance diagram 
    tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())#a
    fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())#b
    fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())#c
    tn = tf.keras.metrics.TrueNegatives(thresholds=thresh.tolist())#d

    day_1_tp = tp(day_1_labels,day_1_output)
    day_1_fp = fp(day_1_labels,day_1_output)
    day_1_fn = fn(day_1_labels,day_1_output)
    day_1_tn = tn(day_1_labels,day_1_output)

    day_1_pod = day_1_tp/(day_1_tp+day_1_fn)
    day_1_srs = day_1_tp/(day_1_tp+day_1_fp)
    day_1_csi = day_1_tp/(day_1_tp+day_1_fn+day_1_fp)

    day_1_precision, day_1_recall, day_1_thresholds = precision_recall_curve(np.ravel(day_1_labels),np.ravel(day_1_output))
    day_1_auc = auc(day_1_recall,day_1_precision)
    print('day_1_auc,',day_1_auc)
    day_1_dict = {'csi':day_1_csi, 'srs':day_1_srs,'pod':day_1_pod,'auc':day_1_auc}

    label = 'Day 1 - Max CSI: %s'%(f"{max(day_1_csi):.2f}")
    print('rotation:',rotation,'UNet')
    print('day 1 max csi:',max(day_1_csi))
    day1_max_idx = np.where(day_1_csi==max(day_1_csi))
    print('day 1 max csi threshold:',thresh[day1_max_idx])
    ax.plot(np.asarray(day_1_srs),np.asarray(day_1_pod),'-s',
                color=colors[0],
                markerfacecolor=colors[0],
                label=label,
                linewidth=3)

    day_2_tp = tp(day_2_labels,day_2_output)
    day_2_fp = fp(day_2_labels,day_2_output)
    day_2_fn = fn(day_2_labels,day_2_output)
    day_2_tn = tn(day_2_labels,day_2_output)

    day_2_pod = day_2_tp/(day_2_tp+day_2_fn)
    day_2_srs = day_2_tp/(day_2_tp+day_2_fp)
    day_2_csi = day_2_tp/(day_2_tp+day_2_fn+day_2_fp)

    day_2_precision, day_2_recall, day_2_thresholds = precision_recall_curve(np.ravel(day_2_labels),np.ravel(day_2_output))
    day_2_auc = auc(day_2_recall,day_2_precision)
    day_2_dict = {'csi':day_2_csi, 'srs':day_2_srs,'pod':day_2_pod,'auc':day_2_auc}

    label = 'Day 2 - Max CSI: %s'%(f"{max(day_2_csi):.2f}")
    print('rotation:',rotation,'UNet')
    print('day 2 max csi:',max(day_2_csi))
    day2_max_idx = np.where(day_2_csi==max(day_2_csi))
    print('day 2 max csi threshold:',thresh[day2_max_idx])
    ax.plot(np.asarray(day_2_srs),np.asarray(day_2_pod),'-s',
                color=colors[1],
                markerfacecolor=colors[1],
                label=label,
                linewidth=3)
    
    day_3_tp = tp(day_3_labels,day_3_output)
    day_3_fp = fp(day_3_labels,day_3_output)
    day_3_fn = fn(day_3_labels,day_3_output)
    day_3_tn = tn(day_3_labels,day_3_output)

    day_3_pod = day_3_tp/(day_3_tp+day_3_fn)
    day_3_srs = day_3_tp/(day_3_tp+day_3_fp)
    day_3_csi = day_3_tp/(day_3_tp+day_3_fn+day_3_fp)

    day_3_precision, day_3_recall, day_3_thresholds = precision_recall_curve(np.ravel(day_3_labels),np.ravel(day_3_output))
    day_3_auc = auc(day_3_recall,day_3_precision)
    day_3_dict = {'csi':day_3_csi, 'srs':day_3_srs,'pod':day_3_pod,'auc':day_3_auc}

    label = 'Day 3 - Max CSI: %s'%(f"{max(day_3_csi):.2f}")
    print('rotation:',rotation,'UNet')
    print('day 3 max csi:',max(day_3_csi))
    day3_max_idx = np.where(day_3_csi==max(day_3_csi))
    print('day 3 max csi threshold:',thresh[day3_max_idx])
    ax.plot(np.asarray(day_3_srs),np.asarray(day_3_pod),'-s',
                color=colors[2],
                markerfacecolor=colors[2],
                label=label,
                linewidth=3)

    day_4_tp = tp(day_4_labels,day_4_output)
    day_4_fp = fp(day_4_labels,day_4_output)
    day_4_fn = fn(day_4_labels,day_4_output)
    day_4_tn = tn(day_4_labels,day_4_output)

    day_4_pod = day_4_tp/(day_4_tp+day_4_fn)
    day_4_srs = day_4_tp/(day_4_tp+day_4_fp)
    day_4_csi = day_4_tp/(day_4_tp+day_4_fn+day_4_fp)

    day_4_precision, day_4_recall, day_4_thresholds = precision_recall_curve(np.ravel(day_4_labels),np.ravel(day_4_output))
    day_4_auc = auc(day_4_recall,day_4_precision)
    day_4_dict = {'csi':day_4_csi, 'srs':day_4_srs,'pod':day_4_pod,'auc':day_4_auc}

    label = 'Day 4 - Max CSI: %s'%(f"{max(day_4_csi):.2f}")
    print('rotation:',rotation,'UNet')
    print('day 4 max csi:',max(day_4_csi))
    day4_max_idx = np.where(day_4_csi==max(day_4_csi))
    print('day 4 max csi threshold:',thresh[day4_max_idx])
    ax.plot(np.asarray(day_4_srs),np.asarray(day_4_pod),'-s',
                color=colors[3],
                markerfacecolor=colors[3],
                label=label,
                linewidth=3)
    plt.legend()
    plt.savefig('./performance_diagrams/daily_gewitter_UNet_rot_%s.pdf'%rotation)
    plt.savefig('./performance_diagrams/daily_gewitter_UNet_rot_%s.png'%rotation)
    plt.close()
    print()
    print()
    all_dict = {'day_1':day_1_dict,'day_2':day_2_dict,'day_3':day_3_dict,'day_4':day_4_dict}
    pickle.dump(all_dict,open('./UNet_daily_dict/daily_dict_rot_%s_unet.pkl'%(rotation),'wb'))

def daily_gewitter_LSTM(rotation=4,
                        lstm_deep=1,
                        conv_deep=1):

    lstm_out_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/labels_outputs/'
    file_load = 'BC_LSTM_rot_%s_conv_4_conv_deep_%s_lstm_deep_%s_no_drop_no_shuffle_model_output_labels.pkl'%(rotation, conv_deep, lstm_deep)

    rot_dict = pickle.load(open(lstm_out_dir+file_load,'rb'))
    model_output = rot_dict['model_output']
    labels = rot_dict['labels']

    day_1_output = model_output[:,0,:,:]
    day_1_labels = labels[:,0,:,:]

    day_2_output = model_output[:,1,:,:]
    day_2_labels = labels[:,1,:,:]

    day_3_output = model_output[:,2,:,:]
    day_3_labels = labels[:,2,:,:]

    day_4_output = model_output[:,3,:,:]
    day_4_labels = labels[:,3,:,:]

    #plot it up  
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    ax = make_performance_diagram_axis(ax)
    colors = ['#eff3ff','#bdd7e7','#6baed6','#2171b5']
    thresh = np.arange(0.05,1.05,0.05)

    #statistics we need for performance diagram 
    tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())#a
    fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())#b
    fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())#c
    tn = tf.keras.metrics.TrueNegatives(thresholds=thresh.tolist())#d

    day_1_tp = tp(day_1_labels,day_1_output)
    day_1_fp = fp(day_1_labels,day_1_output)
    day_1_fn = fn(day_1_labels,day_1_output)
    day_1_tn = tn(day_1_labels,day_1_output)

    day_1_pod = day_1_tp/(day_1_tp+day_1_fn)
    day_1_srs = day_1_tp/(day_1_tp+day_1_fp)
    day_1_csi = day_1_tp/(day_1_tp+day_1_fn+day_1_fp)
    
    day_1_precision, day_1_recall, day_1_thresholds = precision_recall_curve(np.ravel(day_1_labels),np.ravel(day_1_output))
    day_1_auc = auc(day_1_recall,day_1_precision)
    print('day_1_auc,',day_1_auc)
    day_1_dict = {'csi':day_1_csi, 'srs':day_1_srs,'pod':day_1_pod,'auc':day_1_auc}

    label = 'Day 1 - Max CSI: %s'%(f"{max(day_1_csi):.2f}")
    print('rotation:',rotation,'lstm_deep:',lstm_deep,)
    print('day 1 max csi:',max(day_1_csi))
    day1_max_idx = np.where(day_1_csi==max(day_1_csi))
    print('day 1 max csi threshold:',thresh[day1_max_idx])


    ax.plot(np.asarray(day_1_srs),np.asarray(day_1_pod),'-s',
                color=colors[0],
                markerfacecolor=colors[0],
                label=label,
                linewidth=3)

    #  calc x,y of performance diagram 
    #     pods = tps/(tps + fns)
    #     srs = tps/(tps + fps)
    #     csis = tps/(tps + fns + fps)

    day_2_tp = tp(day_2_labels,day_2_output)
    day_2_fp = fp(day_2_labels,day_2_output)
    day_2_fn = fn(day_2_labels,day_2_output)
    day_2_tn = tn(day_2_labels,day_2_output)


    day_2_pod = day_2_tp/(day_2_tp+day_2_fn)
    day_2_srs = day_2_tp/(day_2_tp+day_2_fp)
    day_2_csi = day_2_tp/(day_2_tp+day_2_fn+day_2_fp)

    day_2_precision, day_2_recall, day_2_thresholds = precision_recall_curve(np.ravel(day_2_labels),np.ravel(day_2_output))
    day_2_auc = auc(day_2_recall,day_2_precision)
    print('day_2_auc,',day_2_auc)
    day_2_dict = {'csi':day_2_csi, 'srs':day_2_srs,'pod':day_2_pod,'auc':day_2_auc}

    label = 'Day 2 - Max CSI: %s'%(f"{max(day_2_csi):.2f}")
    print('day 2 max csi:',max(day_2_csi))
    day2_max_idx = np.where(day_2_csi==max(day_2_csi))
    print('day 2 max csi threshold:',thresh[day2_max_idx])
    ax.plot(np.asarray(day_2_srs),np.asarray(day_2_pod),'-s',
                color=colors[1],
                markerfacecolor=colors[1],
                label=label,
                linewidth=3)
    
    day_3_tp = tp(day_3_labels,day_3_output)
    day_3_fp = fp(day_3_labels,day_3_output)
    day_3_fn = fn(day_3_labels,day_3_output)
    day_3_tn = tn(day_3_labels,day_3_output)

    day_3_pod = day_3_tp/(day_3_tp+day_3_fn)
    day_3_srs = day_3_tp/(day_3_tp+day_3_fp)
    day_3_csi = day_3_tp/(day_3_tp+day_3_fn+day_3_fp)
    day_3_precision, day_3_recall, day_3_thresholds = precision_recall_curve(np.ravel(day_3_labels),np.ravel(day_3_output))
    day_3_auc = auc(day_3_recall,day_3_precision)
    print('day_3_auc,',day_3_auc)
    day_3_dict = {'csi':day_3_csi, 'srs':day_3_srs,'pod':day_3_pod,'auc':day_3_auc}

    label = 'Day 3 - Max CSI: %s'%(f"{max(day_3_csi):.2f}")
    print('day 3 max csi:',max(day_3_csi))
    day3_max_idx = np.where(day_3_csi==max(day_3_csi))
    print('day 3 max csi threshold:',thresh[day3_max_idx])
    ax.plot(np.asarray(day_3_srs),np.asarray(day_3_pod),'-s',
                color=colors[2],
                markerfacecolor=colors[2],
                label=label,
                linewidth=3)

    day_4_tp = tp(day_4_labels,day_4_output)
    day_4_fp = fp(day_4_labels,day_4_output)
    day_4_fn = fn(day_4_labels,day_4_output)
    day_4_tn = tn(day_4_labels,day_4_output)

    day_4_pod = day_4_tp/(day_4_tp+day_4_fn)
    day_4_srs = day_4_tp/(day_4_tp+day_4_fp)
    day_4_csi = day_4_tp/(day_4_tp+day_4_fn+day_4_fp)
    day_4_precision, day_4_recall, day_4_thresholds = precision_recall_curve(np.ravel(day_4_labels),np.ravel(day_4_output))
    day_4_auc = auc(day_4_recall,day_4_precision)
    print('day_4_auc,',day_4_auc)
    day_4_dict = {'csi':day_4_csi, 'srs':day_4_srs,'pod':day_4_pod,'auc':day_4_auc}
    

    label = 'Day 4 - Max CSI: %s'%(f"{max(day_4_csi):.2f}")
    print('day 4 max csi:',max(day_4_csi))
    day4_max_idx = np.where(day_4_csi==max(day_4_csi))
    print('day 4 max csi threshold:',thresh[day4_max_idx])
    ax.plot(np.asarray(day_4_srs),np.asarray(day_4_pod),'-s',
                color=colors[3],
                markerfacecolor=colors[3],
                label=label,
                linewidth=3)
    plt.legend()
    plt.savefig('./performance_diagrams/daily_gewitter_LSTM_rot_%s_conv_deep_%s_lstm_deep_%s.pdf'%(rotation,conv_deep,lstm_deep))
    plt.savefig('./performance_diagrams/daily_gewitter_LSTM_rot_%s_conv_deep_%s_lstm_deep_%s.png'%(rotation,conv_deep,lstm_deep))
    plt.close()
    all_dict = {'day_1':day_1_dict,'day_2':day_2_dict,'day_3':day_3_dict,'day_4':day_4_dict}
    pickle.dump(all_dict,open('./daily_dict_rot_%s_conv_deep_%s_lstm_deep_%s.pkl'%(rotation,conv_deep,lstm_deep),'wb'))

def daily_gewitter_calc_AIES():
    print('calculating the daily gewitter curves')
    model_output_types = ['unet_all','lstm_all']
    days = [0,1,2,3]
    data = pickle.load(open('/scratch/bmac87/BC_reviewer_analysis/data.pkl','rb'))
    y_true = data['y_all']
    thresh = np.arange(0.05,1.05,0.05)

    #initialize the arrays to store the statistics
    csis = np.zeros((2,4,len(thresh)))#model, day, thresholds
    aucs = np.zeros((2,4))#model,day
    precision = np.zeros((2,4,len(thresh)))#model, day, thresholds
    recall = np.zeros((2,4,len(thresh)))#model, day, thresholds
    max_csi = np.zeros((2,4))#model, day
    max_csi_thresholds = np.zeros((2,4))#model, day

    #statistics we need for performance diagram 
    tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())#a
    fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())#b
    fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())#c
    tn = tf.keras.metrics.TrueNegatives(thresholds=thresh.tolist())#d

    for m,model in enumerate(model_output_types):
        for d in days:
            daily_pred = data[model][:,d,:,:]
            daily_true = y_true[:,d,:,:]

            daily_tp = tp(daily_true,daily_pred)
            daily_fp = fp(daily_true,daily_pred)
            daily_fn = fn(daily_true,daily_pred)
            daily_tn = tn(daily_true,daily_pred)

            recall[m,d,:] = daily_tp/(daily_tp+daily_fn)
            precision[m,d,:] = daily_tp/(daily_tp+daily_fp)
            csis[m,d,:] = daily_tp/(daily_tp+daily_fn+daily_fp)

            sk_precision, sk_recall, sk_thresholds = precision_recall_curve(y_true=np.ravel(daily_true),y_score=np.ravel(daily_pred))
            aucs[m,d] = auc(sk_recall,sk_precision)
            
            max_csi[m,d] = np.max(csis[m,d,:])
            max_csi_idx = np.where(csis[m,d,:]==max_csi[m,d])[0]
            max_csi_thresholds[m,d]=thresh[max_csi_idx]
            
            print(m,model,d,thresh[max_csi_idx],max_csi[m,d],aucs[m,d])
            del daily_pred, daily_true, daily_tp, daily_fp, daily_fn, daily_tn,max_csi_idx

    stats_dict = {
        'recall':recall,
        'precision':precision,
        'csis':csis,
        'aucs':aucs,
        'max_csi':max_csi,
        'max_csi_thresholds':max_csi_thresholds
    }
    pickle.dump(stats_dict,open('/scratch/bmac87/BC_reviewer_analysis/stats_dict.pkl','wb'))
    del stats_dict

def daily_gewitter_plot_AIES(lstm_deep=1,conv_deep=1):

    #load the datom from daily_gewitter_calc_AIES()
    stats_data = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/BC_reviewer_analysis/stats/stats_dict.pkl','rb'))
    recall = stats_data['recall']
    precision = stats_data['precision']
    auc = stats_data['aucs']
    print(recall.shape)
    print(precision.shape)
    print(auc.shape)
    
    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']  # Colorblind-friendly
    linestyles = ['dashed','dashdot','dotted','solid'] #np.flip(['-', '--', '-.', ':'])
    markers = ['s', 'o', '^', 'D']
    thresh = np.arange(0.05,1.05,0.05)

    #plot it up  
    fig, axes = plt.subplots(1,2,figsize=(30,10))#(x,y)
    unet_ax = make_performance_diagram_axis(axes[0], csi_cmap='Greys')
    lstm_ax = make_performance_diagram_axis(axes[1], csi_cmap='Greys')

    for m in range(2):#model, 0 unet, 1 lstm
        for i in range(4):#loop over days: 0,1,2,3
            label = 'Day %s - AUC: %s'%((i+1),f"{auc[m,i]:.2f}")
            axes[m].plot(precision[m,i,:],recall[m,i,:],
                        color=colors[i],
                        marker=markers[i],
                        markerfacecolor=colors[i],
                        markeredgecolor='black',
                        markersize=10,
                        label=label,
                        linestyle=linestyles[i],
                        linewidth=3)
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel('Recall (POD)')
    axes[0].set_xlabel('Precision (SR)')
    axes[0].set_title('UNet',fontsize=24)
    axes[0].text(.05,.9,'(a)',fontsize=24,fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].set_title('LSTM',fontsize=24)
    axes[1].text(.05,.9,'(b)',fontsize=24,fontweight='bold')
    axes[1].set_ylabel('Recall (POD)')
    axes[1].set_xlabel('Precision (SR)')
    
    for i,t in enumerate(thresh):
        if i%2==0:
            text = np.char.ljust(str(np.round(t+.05,2)),width=4,fillchar='0')
            axes[0].text(precision[0,0,i+1]+0.02,recall[0,0,i+1]+0.01,text,path_effects=pe1,fontsize=18,color=colors[0])
            axes[1].text(precision[1,0,i+1]+0.02,recall[1,0,i+1]+0.01,text,path_effects=pe1,fontsize=18,color=colors[0])

            # if i<=15:
            #     text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
            #     axes[0].text(precision[1,-1,i]-0.06,recall[1,-1,i]-.04,text,path_effects=pe1,fontsize=18,color=colors[-1])
            #     axes[1].text(precision[1,-1,i]-0.05,recall[1,-1,i]-0.04,text,path_effects=pe1,fontsize=18,color=colors[-1])
    plt.savefig('./AIES_results_reviewer_edits/PD.pdf')
    plt.savefig('./AIES_results_reviewer_edits/PD.png')
    plt.tight_layout()
    plt.close()

if __name__=="__main__":
    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print(n_visible_devices)
    tf.config.set_visible_devices([], 'GPU')
    print('GPU turned off')
    daily_gewitter_plot_AIES()
