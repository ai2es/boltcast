import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import shutil
import pickle

def plot_pdp(loc='vance',conv_deep=0,lstm_deep=1):

    #AMS Style dates: 1500 UTC 3 May 2015.
    features = ['cape','precip_rate','reflectivity','lifted_idx','w','graupel_q','ice_q','snow_q','rain_q']
    calc_dir = './AIES_results_reviewer_edits/pdp_calculations_pkl/'
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
    ylabel = 'Probability of Lightning'

    fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(20,10))
    if loc == 'wright_patt':
        title_text = '12 UTC 13-14 June 2022'

    if loc == 'vance':
        title_text = '12 UTC 9-10 Dec 2022'
    #CAPE
    data = pickle.load(open('%s%s_UNet_cape_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_cape_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[0].set_ylim([0,1])
    axes[0].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0].set_ylabel(ylabel,fontsize=24)
    axes[0].set_xticks([0,1500,3000,4500],['0','1500','3000','4500'],fontsize=24)
    axes[0].set_xlabel('CAPE (J/kg)',fontsize=24)
    axes[0].grid('on')
    axes[0].legend(fontsize=18,loc='center right',facecolor='white',framealpha=1)
    axes[0].set_title(title_text,fontsize=24)
    
    #reflectivity
    data = pickle.load(open('%s%s_UNet_reflectivity_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    axes[1].set_ylim([0,1])
    axes[1].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1].grid('on')

    data = pickle.load(open('%s%s_LSTM_reflectivity_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[1].set_xlim([-24,54])
    axes[1].set_xticks([-20,-10,0,10,20,30,40,50],['-20','-10','0','10','20','30','40','50'],fontsize=24)
    axes[1].set_xlabel('Reflectivity (dBZ)',fontsize=24)
    axes[1].set_title(title_text,fontsize=24)

    #UNet, precip_rate
    data = pickle.load(open('%s%s_UNet_precip_rate_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[2].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    axes[2].set_ylim([0,1])
    axes[2].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[2].grid('on')
    
    data = pickle.load(open('%s%s_LSTM_precip_rate_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    
    if loc=='wright_patt':
        axes[2].set_xlim([-.0001,.0057])
        axes[2].set_xticks([0,.002,.004],['0','.002','.004'],fontsize=24)
    if loc=='vance':
        axes[2].set_xlim([-.001, .011])
        axes[2].set_xticks([0,.002,.004,.006,.008,.01],['0','.002','.004','.006','.008','.010'],fontsize=24)
    axes[2].set_xlabel('Precip Rate (kg/(m$^{2}$ s))',fontsize=24)
    axes[2].set_title(title_text,fontsize=24)

    if loc=='wright_patt':
        axes[0].text(5,.15,'(a)',fontsize=24,fontweight='bold')
        axes[1].text(-18,.15,'(b)',fontsize=24,fontweight='bold')
        axes[2].text(.0001,.15,'(c)',fontsize=24,fontweight='bold')
    if loc=='vance':
        axes[0].text(3000,.25,'(a)',fontsize=24,fontweight='bold')
        axes[1].text(40,.25,'(b)',fontsize=24,fontweight='bold')
        axes[2].text(.008,.25,'(c)',fontsize=24,fontweight='bold')

    plt.savefig('./AIES_results_reviewer_edits/pdp_%s.png'%loc)
    plt.savefig('./AIES_results_reviewer_edits/pdp_%s.pdf'%loc)
    plt.close()

def plot_pdp_remainder(conv_deep=0,lstm_deep=1):
    print('plotting the non-top 3 PDP plots for inclusion in supplemental material')

    #AMS Style dates: 1500 UTC 3 May 2015.
    features = ['lifted_idx','w','graupel_q','ice_q','snow_q','rain_q']
    calc_dir = './AIES_results_reviewer_edits/pdp_calculations_pkl/'
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

    fig,axes = plt.subplots(nrows=2,ncols=6,figsize=(50,30))
    loc = 'wright_patt'
    title_text = '12 UTC 13-14 June 2022'
    #lifted index
    data = pickle.load(open('%s%s_UNet_lifted_idx_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0,0].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0,0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0,0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[0,0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[0,0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_lifted_idx_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0,0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0,0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[0,0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[0,0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[0,0].set_ylim([0,1])
    axes[0,0].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0,0].set_ylabel('Wright-Patterson AFB Lightning Probability',fontsize=24)
    axes[0,0].set_xticks([-8,0,10,20,30,35],['-8','0','10','20','30','35'],fontsize=24)
    axes[0,0].grid('on')
    axes[0,0].legend(fontsize=18,loc='center right',facecolor='white',framealpha=1)
    axes[0,0].set_title(title_text,fontsize=18)

    #vertical velocity
    data = pickle.load(open('%s%s_UNet_w_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0,1].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0,1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0,1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[0,1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[0,1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_w_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0,1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0,1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[0,1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[0,1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[0,1].set_ylim([0,1])
    axes[0,1].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0,1].set_xticks([0,.5,1,1.5,2.0,2.5,3.0],['0','0.5','1.0','1.5','2.0','2.5','3.0'],fontsize=24)
    axes[0,1].grid('on')
    axes[0,1].set_title(title_text,fontsize=18)

    #graupel mixing ratio
    data = pickle.load(open('%s%s_UNet_graupel_q_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0,2].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0,2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0,2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[0,2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[0,2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_graupel_q_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0,2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0,2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[0,2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[0,2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[0,2].set_ylim([0,1])
    axes[0,2].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0,2].set_xticks([0,.0004,.0008,.0012,.0016,.002],['0','4','8','12','16','20'],fontsize=24)
    axes[0,2].grid('on')
    axes[0,2].set_title(title_text,fontsize=18)

    #ice mixing ratio
    data = pickle.load(open('%s%s_UNet_ice_q_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0,3].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0,3].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0,3].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[0,3].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[0,3].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_ice_q_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0,3].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0,3].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[0,3].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[0,3].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[0,3].set_ylim([0,1])
    axes[0,3].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0,3].set_xticks([0,.0002,.0004,.0006,.0008,.0010],['0','2','4','6','8','10'],fontsize=24)
    axes[0,3].grid('on')
    axes[0,3].set_title(title_text,fontsize=18)

    #snow mixing ratio
    data = pickle.load(open('%s%s_UNet_snow_q_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0,4].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0,4].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0,4].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[0,4].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[0,4].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_snow_q_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0,4].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0,4].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[0,4].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[0,4].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[0,4].set_ylim([0,1])
    axes[0,4].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0,4].set_xticks([0,.0005,.0010,.0015,.0020,.0025,.0030,.0035],['0','5','10','15','20','25','30','35'],fontsize=24)
    axes[0,4].grid('on')
    axes[0,4].set_title(title_text,fontsize=18)

    #rain mixing ratio
    data = pickle.load(open('%s%s_UNet_rain_q_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[0,5].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[0,5].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[0,5].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[0,5].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[0,5].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_rain_q_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[0,5].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[0,5].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[0,5].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[0,5].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[0,5].set_ylim([0,1])
    axes[0,5].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[0,5].set_xticks([0,.0002,.0004,.0006,.0008,.0010,.0012,.0014],['0','2','4','6','8','10','12','14'],fontsize=24)
    axes[0,5].grid('on')
    axes[0,5].set_title(title_text,fontsize=18)

    loc = 'vance'
    title_text = '12 UTC 9-10 Dec 2022'
    #lifted index
    data = pickle.load(open('%s%s_UNet_lifted_idx_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,0].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1,0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1,0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[1,0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[1,0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_lifted_idx_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[1,0].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1,0].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[1,0].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[1,0].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[1,0].set_ylim([0,1])
    axes[1,0].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1,0].set_ylabel('Vance AFB Lightning Probability',fontsize=24)
    axes[1,0].set_xticks([-8,0,10,20,30,35],['-8','0','10','20','30','35'],fontsize=24)
    axes[1,0].grid('on')
    axes[1,0].set_title(title_text,fontsize=18)

    #vertical velocity
    data = pickle.load(open('%s%s_UNet_w_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,1].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1,1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1,1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[1,1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[1,1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_w_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[1,1].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1,1].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[1,1].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[1,1].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[1,1].set_ylim([0,1])
    axes[1,1].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1,1].set_xticks([0,1,2,3,4,5,6,7],['0','1.0','2.0','3.0','4.0','5.0','6.0','7.0'],fontsize=24)
    axes[1,1].grid('on')
    axes[1,1].set_title(title_text,fontsize=18)

    #graupel mixing ratio
    data = pickle.load(open('%s%s_UNet_graupel_q_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,2].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1,2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1,2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[1,2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[1,2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_graupel_q_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[1,2].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1,2].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[1,2].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[1,2].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[1,2].set_ylim([0,1])
    axes[1,2].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1,2].set_xticks([0,.0004,.0008,.0012,.0016,.002],['0','4','8','12','16','20'],fontsize=24)
    axes[1,2].grid('on')
    axes[1,2].set_title(title_text,fontsize=18)

    #ice mixing ratio
    data = pickle.load(open('%s%s_UNet_ice_q_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,3].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1,3].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1,3].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[1,3].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[1,3].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_ice_q_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[1,3].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1,3].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[1,3].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[1,3].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[1,3].set_ylim([0,1])
    axes[1,3].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1,3].set_xticks([0,.0002,.0004,.0006,.0008,.0010,.0012],['0','2','4','6','8','10','12'],fontsize=24)
    axes[1,3].grid('on')
    axes[1,3].set_title(title_text,fontsize=18)

    #snow mixing ratio
    data = pickle.load(open('%s%s_UNet_snow_q_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,4].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1,4].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1,4].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[1,4].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[1,4].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_snow_q_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[1,4].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1,4].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[1,4].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[1,4].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[1,4].set_ylim([0,1])
    axes[1,4].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1,4].set_xticks([0,.0010,.0020,.0030,.0040,.0050,.0060,.0070,.0080],['0','10','20','30','40','50','60','70','80'],fontsize=24)
    axes[1,4].grid('on')
    axes[1,4].set_title(title_text,fontsize=18)

    #rain mixing ratio
    data = pickle.load(open('%s%s_UNet_rain_q_pred_dict.pkl'%(calc_dir,loc),'rb'))
    preds_array = data['preds_array']
    feature_values = data['feature_values']
    axes[1,5].vlines(feature_values,ymin=0,ymax=.1,color='black',linewidth=1)
    axes[1,5].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=unet_line,linewidth=lw,label='UNet')
    axes[1,5].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=unet_line,linewidth=lw,label='4 grid points')
    axes[1,5].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=unet_line,linewidth=lw,label='8 grid points')
    axes[1,5].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=unet_line,linewidth=lw,label='12 grid points')
    
    data = pickle.load(open('%s%s_LSTM_rain_q_%s_conv_deep_%s_lstm_deep_pred_dict.pkl'%(calc_dir,loc,conv_deep,lstm_deep),'rb'))
    preds_array = data['preds_array']
    axes[1,5].plot(feature_values,preds_array[:,0],color=colors[0],linestyle=lstm_line,linewidth=lw,label='LSTM')
    axes[1,5].plot(feature_values,preds_array[:,1],color=colors[1],linestyle=lstm_line,linewidth=lw,label='4 grid points')
    axes[1,5].plot(feature_values,preds_array[:,2],color=colors[2],linestyle=lstm_line,linewidth=lw,label='8 grid points')
    axes[1,5].plot(feature_values,preds_array[:,3],color=colors[3],linestyle=lstm_line,linewidth=lw,label='12 grid points')
    axes[1,5].set_ylim([0,1])
    axes[1,5].set_yticks([0,.2,.4,.6,.8,1],['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=24)
    axes[1,5].set_xticks([0,.0002,.0004,.0006,.0008,.0010,.0012,.0014],['0','2','4','6','8','10','12','14'],fontsize=24)
    axes[1,5].grid('on')
    axes[1,5].set_title(title_text,fontsize=18)

    axes[1,0].set_xlabel('Lifted Index (K)',fontsize=24)
    axes[1,1].set_xlabel('Vertical Velocity (m/s)',fontsize=24)
    axes[1,2].set_xlabel('Graupel Mixing Ratio (kg/kg) (1e4)',fontsize=24)
    axes[1,3].set_xlabel('Ice Mixing Ratio (kg/kg) (1e3)',fontsize=24)
    axes[1,4].set_xlabel('Snow Mixing Ratio (kg/kg) (1e4)',fontsize=24)
    axes[1,5].set_xlabel('Rain Mixing Ratio (kg/kg) (1e4)',fontsize=24)

    axes[0,0].text(-6,.15,'(a)',fontsize=24,fontweight='bold')
    axes[0,1].text(.15,.15,'(b)',fontsize=24,fontweight='bold')
    axes[0,2].text(.00015,.15,'(c)',fontsize=24,fontweight='bold')
    axes[0,3].text(.00005,.15,'(d)',fontsize=24,fontweight='bold')
    axes[0,4].text(.00005,.15,'(e)',fontsize=24,fontweight='bold')
    axes[0,5].text(.00005,.15,'(f)',fontsize=24,fontweight='bold')

    axes[1,0].text(36,.15,'(g)',fontsize=24,fontweight='bold')
    axes[1,1].text(6.1,.15,'(h)',fontsize=24,fontweight='bold')
    axes[1,2].text(.0022,.15,'(i)',fontsize=24,fontweight='bold')
    axes[1,3].text(.00105,.15,'(j)',fontsize=24,fontweight='bold')
    axes[1,4].text(.00705,.15,'(k)',fontsize=24,fontweight='bold')
    axes[1,5].text(.00122,.15,'(l)',fontsize=24,fontweight='bold')

    plt.savefig('./AIES_results_reviewer_edits/pdp_remainder.png')
    plt.savefig('./AIES_results_reviewer_edits/pdp_remainder.pdf')
    plt.close()

if __name__=='__main__':
    plot_pdp(loc='vance')
    plot_pdp(loc='wright_patt')
    plot_pdp_remainder()



