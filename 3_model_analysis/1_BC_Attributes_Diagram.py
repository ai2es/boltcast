import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
import matplotlib.patheffects as path_effects

def BoltCast_attributes_plot(load_dir='/scratch/bmac87/BC_reviewer_analysis_v2/'):
    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']  # Colorblind-friendly
    linestyles = ['dashed','dashdot','dotted','solid'] #np.flip(['-', '--', '-.', ':'])
    markers = ['s', 'o', '^', 'D']
    days = ['1','2','3','4']
    perfect = np.linspace(0,1,100)
    pe1 = [path_effects.withStroke(linewidth=1.5,foreground="k")]
    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(20,20))
    climos = []
    for d,day in enumerate(days):
        data = pickle.load(open('%sDay_attr_%s.pkl'%(load_dir,day),'rb'))
        unet_data = data['unet']
        lstm_data = data['lstm']
        climos.append(unet_data['climo'])
        
        x_unet = unet_data['unet_pred']
        y_unet = unet_data['unet_true']
        bs_unet = unet_data['unet_bs']
        axes[0,0].plot(x_unet[x_unet>0],y_unet[y_unet>0],linewidth=5.0,marker=markers[d],linestyle=linestyles[d],markeredgecolor='black',markersize=10.0,color=colors[d],label='Day %s - BS = %s'%(day,f"{bs_unet:.03f}"),zorder=3)
        
        x_lstm = lstm_data['lstm_pred']
        y_lstm = lstm_data['lstm_true']
        bs_lstm = lstm_data['lstm_bs']
        axes[0,1].plot(x_lstm[x_lstm>0],y_lstm[y_lstm>0],linewidth=5.0,marker=markers[d],linestyle=linestyles[d],markeredgecolor='black',markersize=10.0,color=colors[d],label='Day %s - BS = %s'%(day,f"{bs_lstm:.03f}"),zorder=3)
    
    climo = np.ones(100)*np.mean(climos)
    no_skill = (climo+perfect)/2
    axes[0,0].plot(perfect,perfect,linestyle='--',color='grey',linewidth=3.0,label='Perfect')
    axes[0,0].set_xlim([0,1])
    axes[0,0].set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[0,0].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],fontsize=18)
    axes[0,0].set_ylim([0,1])
    axes[0,0].set_ylabel('Observed Relative Frequency',fontsize=18)
    axes[0,0].grid(True)
    axes[0,0].set_title('U-Net',fontsize=24)
    axes[0,0].plot(np.mean(climos)*np.ones(100),np.linspace(0,1,100),linestyle='dashdot',linewidth=3.0,color='grey',label='Climatology',zorder=0)
    axes[0,0].plot(np.linspace(0,1,100),np.mean(climos)*np.ones(100),linestyle='dashdot',linewidth=3.0,color='grey',zorder=0)
    axes[0,0].plot(perfect,no_skill,linestyle='solid',linewidth=5.0,color='grey',label='No Skill',zorder=0)
    axes[0,0].legend(fontsize=18,loc='upper left', bbox_to_anchor=(0.11, 0.999))
    axes[0,0].text(.015,.93,'(a)',fontsize=24,weight='heavy',path_effects=pe1)

    axes[0,1].plot(perfect,perfect,linestyle='--',color='grey',linewidth=3.0)
    axes[0,1].set_xlim([0,1])
    axes[0,1].set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[0,1].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],fontsize=18)
    axes[0,1].set_ylim([0,1])
    axes[0,1].plot(np.mean(climos)*np.ones(100),np.linspace(0,1,100),linestyle='dashdot',linewidth=3.0,color='grey',zorder=0)
    axes[0,1].plot(np.linspace(0,1,100),np.mean(climos)*np.ones(100),linestyle='dashdot',linewidth=3.0,color='grey',zorder=0)
    axes[0,1].plot(perfect,no_skill,linestyle='solid',linewidth=5.0,color='grey',zorder=0)
    axes[0,1].grid(True)
    axes[0,1].legend(fontsize=18,loc='upper left', bbox_to_anchor=(0.11, 0.999))
    axes[0,1].set_title('LSTM',fontsize=24)
    axes[0,1].text(.015,.93,'(b)',fontsize=24,weight='heavy',path_effects=pe1)

    thresh = np.arange(0.00,1.05,0.05)
    thresh_centers =  0.5 * (thresh[:-1] + thresh[1:])
    unet_total_counts = np.zeros((4,21))
    for day in range(4):
        temp_data = pickle.load(open('%scounts_day_%s.pkl'%(load_dir,day+1),'rb'))
        unet_total_counts[day,:]=temp_data['unet_thresh_counts']
        del temp_data

    lstm_total_counts = np.zeros((4,21))
    for day in range(4):
        temp_data = pickle.load(open('%scounts_day_%s.pkl'%(load_dir,day+1),'rb'))
        lstm_total_counts[day,:]=temp_data['lstm_thresh_counts']
        del temp_data

    axes[1,0].grid(True,zorder=0)
    axes[1,0].bar(thresh-.025,np.sum(unet_total_counts,axis=0),width=0.05, color='grey', edgecolor='black',zorder=3)
    axes[1,0].set_yscale('log')
    axes[1,0].set_xlim([0,1])
    axes[1,0].set_ylabel('Count',fontsize=18)
    axes[1,0].set_xticks(thresh[::2],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[1,0].set_yticks([ 10**(4),10.0**(5),10.0**(6),10.0**(7),10.0**(8),10.0**(9)],['$10^{4}$','$10^{5}$','$10^{6}$','$10^{7}$','$10^{8}$','$10^{9}$'],fontsize=18)
    axes[1,0].set_xlabel('Forecast Probability',fontsize=18)
    axes[1,0].text(.915,10**8.7,'(c)',fontsize=24,weight='heavy',path_effects=pe1)

    axes[1,1].grid(True,zorder=0)
    axes[1,1].bar(thresh-.025,np.sum(lstm_total_counts,axis=0),width=0.05, color='grey', edgecolor='black',zorder=3)
    axes[1,1].set_yscale('log')
    axes[1,1].set_xlim([0,1])
    axes[1,1].set_xticks(thresh[::2],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[1,1].set_yticks([ 10**(4),10.0**(5),10.0**(6),10.0**(7),10.0**(8),10.0**(9)],['$10^{4}$','$10^{5}$','$10^{6}$','$10^{7}$','$10^{8}$','$10^{9}$'],fontsize=18)
    axes[1,1].set_ylim([10**4,10**9])
    axes[1,1].set_xlabel('Forecast Probability',fontsize=18)
    axes[1,1].text(.915,10**8.7,'(d)',fontsize=24,weight='heavy',path_effects=pe1)

    plt.show()
    plt.savefig('./AIES_results_reviewer_edits_v2/Attributes_11Jan26.png')
    plt.savefig('./AIES_results_reviewer_edits_v2/Attributes_11Jan26.pdf')
    plt.close()

def BoltCast_pred_hist_calc_np(save_dir = '/scratch/bmac87/BC_reviewer_analysis_v2/', y_true=[], y_pred_lstm=[],y_pred_unet=[]):
    print('generating the BC attributes diagram with numpy for better control')
    thresh = np.arange(0.00,1.05,0.05)
    print(thresh)
    thresh_centers =  0.5 * (thresh[:-1] + thresh[1:])

    for day in range(4):
        y_day_true = np.ravel(y_true[:,day,:,:])
        y_lstm = np.ravel(y_pred_lstm[:,day,:,:])
        y_unet = np.ravel(y_pred_unet[:,day,:,:])

        #bin the predictions into the thresholds. inclusive to the right
        lstm_indices = np.digitize(y_lstm,bins=thresh,right=True)
        unet_indices = np.digitize(y_unet,bins=thresh,right=True)

        # Initialize lists
        thresh_lstm_counts = np.zeros(len(thresh))
        thresh_unet_counts = np.zeros(len(thresh))

        # Compute stats for each bin
        for i in range(1, len(thresh)):
            lstm_idx = lstm_indices == i
            unet_idx = unet_indices == i
            if np.sum(lstm_idx)>0:
                thresh_lstm_counts[i] = np.sum(lstm_idx)

            if np.sum(unet_idx)>0:
                thresh_unet_counts[i] = np.sum(unet_idx)
        pickle.dump({'unet_thresh_counts':thresh_unet_counts,'lstm_thresh_counts':thresh_lstm_counts},open('%scounts_day_%s.pkl'%(save_dir,day+1),'wb'))

def BoltCast_attributes_calc_np(save_dir = '/scratch/bmac87/BC_reviewer_analysis_v2/',y_true=[],y_pred_lstm=[],y_pred_unet=[]):
    
    print('generating the BC attributes diagram with numpy for better control')
    thresh = np.arange(0.00,1.05,0.05)
    print(thresh)
    thresh_centers =  0.5 * (thresh[:-1] + thresh[1:])

    for day in range(4):
        y_day_true = np.ravel(y_true[:,day,:,:])
        y_lstm = np.ravel(y_pred_lstm[:,day,:,:])
        y_unet = np.ravel(y_pred_unet[:,day,:,:])

        #bin the predictions into the thresholds. inclusive to the right
        lstm_indices = np.digitize(y_lstm,bins=thresh,right=True)
        unet_indices = np.digitize(y_unet,bins=thresh,right=True)

        # Initialize lists
        thresh_lstm_true = np.zeros(len(thresh))
        thresh_lstm_pred = np.zeros(len(thresh))
        thresh_lstm_counts = np.zeros(len(thresh))

        thresh_unet_true = np.zeros(len(thresh))
        thresh_unet_pred = np.zeros(len(thresh))
        thresh_unet_counts = np.zeros(len(thresh))

        # Compute stats for each bin
        for i in range(1, len(thresh)):
            lstm_idx = lstm_indices == i
            unet_idx = unet_indices == i
            if np.sum(lstm_idx)>0:
                thresh_lstm_true[i] = np.mean(y_day_true[lstm_idx])
                thresh_lstm_pred[i] = np.mean(y_lstm[lstm_idx])
                thresh_lstm_counts[i] = np.sum(lstm_idx)

            if np.sum(unet_idx)>0:
                thresh_unet_true[i] = np.mean(y_day_true[unet_idx])
                thresh_unet_pred[i] = np.mean(y_unet[unet_idx])
                thresh_unet_counts[i] = np.sum(unet_idx)
        
        daily_climatology = np.mean(y_day_true)
        daily_unet_brier_score = np.mean((y_unet-y_day_true)**2)
        daily_lstm_brier_score = np.mean((y_lstm-y_day_true)**2)
        daily_brier_score_climatology = np.mean((y_day_true-daily_climatology)**2)
        daily_bss_unet = 1 - (daily_unet_brier_score / daily_brier_score_climatology)
        daily_bss_lstm = 1 - (daily_lstm_brier_score / daily_brier_score_climatology)
        unet_dict = {'unet_true':thresh_unet_true,'unet_pred':thresh_unet_pred,'unet_counts':thresh_unet_counts,'unet_bs':daily_unet_brier_score,'unet_bss':daily_bss_unet,'climo':daily_climatology}
        lstm_dict = {'lstm_true':thresh_lstm_true,'lstm_pred':thresh_lstm_pred,'lstm_counts':thresh_lstm_counts,'lstm_bs':daily_lstm_brier_score,'lstm_bss':daily_bss_lstm,'climo':daily_climatology}
        pickle.dump({'unet':unet_dict,'lstm':lstm_dict},open('%s/Day_attr_%s.pkl'%(save_dir,day+1),'wb'))
        
        del y_day_true, y_lstm, y_unet, lstm_indices, unet_indices
        del thresh_lstm_true, thresh_lstm_pred, thresh_lstm_counts
        del thresh_unet_true, thresh_unet_pred, thresh_unet_counts
        del daily_climatology, daily_unet_brier_score, daily_lstm_brier_score
        del daily_brier_score_climatology
        del daily_bss_unet, daily_bss_lstm, unet_dict, lstm_dict

def save_off_data(save_dir = '/scratch/bmac87/BC_reviewer_analysis_v2/'):
    model_type = 'UNet'
    rots = [0,1,2,3,4]
    true_list = []
    unet_pred_list = []
    proj_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AIES_reviews_v2/'
    fload = 'output_labels.pkl'
    for rot in rots:
        exp_name = 'BC_%s_rot_%s_lrate_0.000010000_'%(model_type,rot)
        load_dir = '%s%s/'%(proj_dir,exp_name)
        y_dict = pickle.load(open(load_dir+fload,'rb'))
        for key in y_dict:
            print(key)
        y_true = y_dict['labels']
        y_pred = y_dict['model_output']
        if rot==0:
            y_true_all = y_true
            unet_pred_all = y_pred
        else:
            y_true_all = np.concatenate([y_true_all,y_true],axis=0)
            unet_pred_all = np.concatenate([unet_pred_all,y_pred],axis=0)
        del y_true, y_pred

    model_type = 'LSTM'
    lstm_pred_list = []
    for rot in rots:
        exp_name = 'BC_%s_rot_%s_lrate_0.000010000_'%(model_type,rot)
        load_dir = '%s%s/'%(proj_dir,exp_name)
        y_dict = pickle.load(open(load_dir+fload,'rb'))
        y_pred = y_dict['model_output']
        if rot==0:
            lstm_pred_all = y_pred
        else:
            lstm_pred_all = np.concatenate([lstm_pred_all,y_pred],axis=0)
    print(lstm_pred_all.shape)
    print(unet_pred_all.shape)
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    pickle.dump({'lstm_all':lstm_pred_all,'unet_all':unet_pred_all,'y_all':y_true_all},open(save_dir+'data.pkl','wb'))

if __name__=='__main__':
    # tf.config.set_visible_devices([], 'GPU')
    # print('NO VISIBLE DEVICES!!!!')
    save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/BC_edits_v2/BC_reviewer_analysis_v2/'
    # save_off_data(save_dir=save_dir)
    # data_dict = pickle.load(open(save_dir+'data.pkl','rb'))
    # y_true = data_dict['y_all']
    # y_pred_unet = data_dict['unet_all']
    # y_pred_lstm = data_dict['lstm_all']
    # BoltCast_attributes_calc_np(save_dir=save_dir,y_true=y_true,y_pred_lstm=y_pred_lstm,y_pred_unet=y_pred_unet)
    # BoltCast_pred_hist_calc_np(save_dir=save_dir,y_true=y_true,y_pred_lstm=y_pred_lstm,y_pred_unet=y_pred_unet)
    BoltCast_attributes_plot(load_dir=save_dir)