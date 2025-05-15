import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import shutil
import pickle

def plot_pdp(loc='vance',
            model_type='UNet',
            conv_deep=0,
            lstm_deep=1):

    #AMS Style dates: 1500 UTC 3 May 2015.
    features = ['cape','precip_rate','reflectivity','lifted_idx','w','graupel_q','ice_q','snow_q','rain_q']

    x_label_dict = {'cape':'CAPE (J/kg)',
        'precip_rate':'Precipitation Rate (kg/(m$^{2}$ s))',
        'reflectivity':'Reflectivity (dBZ)',
        'lifted_idx':'Lifted Index (K)',
        'w':'Vertical Velocity (m/s)',
        'graupel_q':'Graupel Mixing Ratio (kg/kg)',
        'ice_q':'Ice Mixing Ratio (kg/kg)',
        'snow_q':'Snow Mixing Ratio (kg/kg)',
        'rain_q':'Rain Mixing Ratio (kg/kg)'}
    
    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']
    linestyles = ['dashed','dashdot','dotted','solid']
    lw = 3
    unet_line = 'solid'
    lstm_line = 'dotted'

    fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(20,20))
    loc = 'wright_patt'
    title_text = '12 UTC 13-14 June 2022'
    #CAPE
    data = pickle.load(open('./pdps/%s_UNet_cape_pred_dict.pkl'%loc,'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0,0].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0,0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0,0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='1 kernel')
    axes[0,0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='2 kernels')
    axes[0,0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='3 kernels')
    
    data = pickle.load(open('./pdps/%s_LSTM_cape_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0,0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0,0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='1 kernel')
    axes[0,0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='2 kernels')
    axes[0,0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='3 kernels')
    axes[0,0].set_ylim([0,1])
    axes[0,0].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0,0].set_ylabel('Wright-Patterson AFB: Lightning (%)',fontsize=24)
    axes[0,0].set_xticks([0,1500,3000,4500],['0','1500','3000','4500'],fontsize=24)
    axes[0,0].grid('on')
    axes[0,0].legend(fontsize=18,loc='center right',facecolor='white',framealpha=1)
    axes[0,0].set_title(title_text,fontsize=24)
    
    #reflectivity
    data = pickle.load(open('./pdps/%s_UNet_reflectivity_pred_dict.pkl'%loc,'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0,1].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0,1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0,1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='1 kernel')
    axes[0,1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='2 kernels')
    axes[0,1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='3 kernels')
    axes[0,1].set_ylim([0,1])
    axes[0,1].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0,1].grid('on')

    data = pickle.load(open('./pdps/%s_LSTM_reflectivity_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0,1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0,1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='1 kernel')
    axes[0,1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='2 kernels')
    axes[0,1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='3 kernels')
    axes[0,1].set_xlim([-24,54])
    axes[0,1].set_xticks([-20,-10,0,10,20,30,40,50],['-20','-10','0','10','20','30','40','50'],fontsize=24)
    axes[0,1].set_title(title_text,fontsize=24)

    #UNet, precip_rate
    data = pickle.load(open('./pdps/%s_UNet_precip_rate_pred_dict.pkl'%loc,'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0,2].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0,2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0,2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='1 kernel')
    axes[0,2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='2 kernels')
    axes[0,2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='3 kernels')
    axes[0,2].set_ylim([0,1])
    axes[0,2].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0,2].grid('on')
    
    data = pickle.load(open('./pdps/%s_LSTM_precip_rate_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0,2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0,2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='1 kernel')
    axes[0,2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='2 kernels')
    axes[0,2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='3 kernels')
    axes[0,2].set_xlim([-.0001, .006])
    axes[0,2].set_xticks([0,.002,.004,.006],['0','.002','.004','.006'],fontsize=24)
    axes[0,2].set_title(title_text,fontsize=24)

    axes[0,0].text(5,.15,'(a)',fontsize=24,fontweight='bold')
    axes[0,1].text(-18,.15,'(b)',fontsize=24,fontweight='bold')
    axes[0,2].text(.0001,.15,'(c)',fontsize=24,fontweight='bold')
    loc='vance'
    title_text = '12 UTC 9-10 Dec 2022'
    data = pickle.load(open('./pdps/%s_UNet_cape_pred_dict.pkl'%loc,'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,0].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1,0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1,0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='1 kernel')
    axes[1,0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='2 kernels')
    axes[1,0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='3 kernels')

    data = pickle.load(open('./pdps/%s_LSTM_cape_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1,0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='1 kernel')
    axes[1,0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='2 kernels')
    axes[1,0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='3 kernels')
    axes[1,0].set_xlabel(x_label_dict['cape'],fontsize=24)
    axes[1,0].set_ylim([0,1])
    axes[1,0].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1,0].set_ylabel('Vance AFB: Lightning (%)',fontsize=24)
    axes[1,0].set_xticks([0,1500,3000,4500],['0','1500','3000','4500'],fontsize=24)
    axes[1,0].set_title(title_text,fontsize=24)
    axes[1,0].grid('on')

    data = pickle.load(open('./pdps/%s_UNet_reflectivity_pred_dict.pkl'%loc,'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,1].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1,1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1,1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='1 kernel')
    axes[1,1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='2 kernels')
    axes[1,1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='3 kernels')
    
    data = pickle.load(open('./pdps/%s_LSTM_reflectivity_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1,1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='1 kernel')
    axes[1,1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='2 kernels')
    axes[1,1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='3 kernels')
    axes[1,1].set_xlabel(x_label_dict['reflectivity'],fontsize=24)
    axes[1,1].set_ylim([0,1])
    axes[1,1].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1,1].set_xticks([-20,-10,0,10,20,30,40,50],['-20','-10','0','10','20','30','40','50'],fontsize=24)
    axes[1,1].set_title(title_text,fontsize=24)
    axes[1,1].grid('on')

    data = pickle.load(open('./pdps/%s_UNet_precip_rate_pred_dict.pkl'%loc,'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,2].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1,2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1,2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='1 kernel')
    axes[1,2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='2 kernels')
    axes[1,2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='3 kernels')
    axes[1,2].set_xlabel(x_label_dict['precip_rate'],fontsize=24)

    data = pickle.load(open('./pdps/%s_LSTM_precip_rate_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[1,2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1,2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='1 kernel')
    axes[1,2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='2 kernels')
    axes[1,2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='3 kernels')
    axes[1,2].set_ylim([0,1])
    axes[1,2].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1,2].set_xlim([-.0001,0.01])
    axes[1,2].set_xticks([0,.004,.008],['0','0.004','0.008'],fontsize=24)
    axes[1,2].grid('on')
    axes[1,2].set_title(title_text,fontsize=24)

    axes[1,0].text(3800,.15,'(d)',fontsize=24,fontweight='bold')
    axes[1,1].text(40,.15,'(e)',fontsize=24,fontweight='bold')
    axes[1,2].text(.0085,.15,'(f)',fontsize=24,fontweight='bold')

    plt.savefig('./pdps/pdp_summer_winter.png')
    plt.savefig('./pdps/pdp_summer_winter.pdf')
    plt.close()

if __name__=='__main__':
    plot_pdp()



