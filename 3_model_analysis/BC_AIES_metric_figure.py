import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import shutil
import glob

def generate_plot():
    print('generating the plot')
    results_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/model_analysis_output_24Feb25/dict_csi_results/'
    rotations = ['0','1','2','3','4']
    unet_csis = np.zeros((5,4))#rotations, day
    lstm_csis = np.zeros((5,4))#rotations, day
    for r,rot in enumerate(rotations):
        unet_file = 'daily_dict_rot_%s_unet.pkl'%(rot)
        lstm_file = 'daily_dict_rot_%s_lstm_deep_1.pkl'%(rot)
        
        unet_results = pickle.load(open(results_dir+unet_file,'rb'))
        lstm_results = pickle.load(open(results_dir+lstm_file,'rb'))

        unet_csis[r,0] = np.max(unet_results['day_1']['csi'])
        unet_csis[r,1] = np.max(unet_results['day_2']['csi'])
        unet_csis[r,2] = np.max(unet_results['day_3']['csi'])
        unet_csis[r,3] = np.max(unet_results['day_4']['csi'])

        lstm_csis[r,0] = np.max(lstm_results['day_1']['csi'])
        lstm_csis[r,1] = np.max(lstm_results['day_2']['csi'])
        lstm_csis[r,2] = np.max(lstm_results['day_3']['csi'])
        lstm_csis[r,3] = np.max(lstm_results['day_4']['csi'])
    
    matplotlib.rcParams['axes.facecolor'] = [0.8,0.8,0.8] 
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    colors_hex_unet = np.flip(['#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'])
    colors_hex_lstm = np.flip(['#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b'])
    width = .4
    xticks = np.arange(4)
    ax.set_xlabel('Lead Time (# Day)',fontsize=18)
    ax.set_xticks(xticks,['Day 1','Day 2','Day 3','Day 4'],fontsize=18)
    ax.set_title('CSIs versus Lead Time',fontsize=18)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.bar(x=xticks+width/2,height=np.mean(unet_csis,axis=0),width=width,label='UNet',color=colors_hex_unet)
    ax.bar(x=xticks-width/2,height=np.mean(lstm_csis,axis=0),width=width,label='LSTM',color=colors_hex_lstm)
    ax.legend(fontsize=18,facecolor='white')
    ax.set_ylabel('Max. CSI',fontsize=18)
    ax.set_ylim([.3,.44])
    ax.set_yticks([.30,.32,.34,.36,.38,.4,.42,.44],['0.30','0.32','0.34','0.36','0.38','0.40','0.42','0.44'],fontsize=18)
    for xtick in xticks:
        ax.text(xtick,np.mean(unet_csis[:,xtick])+.001,f"{np.mean(unet_csis[:,xtick]):.3f}",fontsize=18)
        ax.text(xtick-width,np.mean(lstm_csis[:,xtick]+.001),f"{np.mean(lstm_csis[:,xtick]):.3f}",fontsize=18)
    plt.savefig('./csi_lead_time.pdf')
    plt.close()

def main():
    print('generating csi figure')
    generate_plot()

if __name__=='__main__':
    main()