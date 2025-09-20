import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, recall_score, precision_score
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_ind
import matplotlib
import matplotlib.patheffects as path_effects

def get_labels_outputs_all(model_type='UNet'):
    data = pickle.load(open('/scratch/bmac87/BC_reviewer_analysis/data.pkl','rb'))
    for key in data:
        print(key)
    if model_type=='UNet':
        y_pred = data['unet_all']
    else:
        y_pred = data['lstm_all']
    y_true = data['y_all']
    return y_true, y_pred

def get_labels_outputs_rot(rot=4,model_type='UNet'):
    print('loading the data')
    if model_type=='UNet':
        output_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/UNet/labels_outputs/'
        fname = 'UNet_symmetric_rot_%s_output.pkl'%(rot)

    if model_type=='LSTM':
        output_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/labels_outputs/'
        fname = 'BC_LSTM_rot_%s_conv_4_conv_deep_0_lstm_deep_1_no_drop_no_shuffle_model_output_labels.pkl'%(rot)
    labels_outputs = pickle.load(open(output_dir+fname,'rb'))
    for key in labels_outputs:
        print(key)
    y_true = labels_outputs['labels']
    print(y_true.shape)
    y_pred = np.squeeze(labels_outputs['model_output'])
    print(y_pred.shape)
    return y_true, y_pred

def calc_auc_gridpts(y_true=np.zeros((10,10)),y_pred=np.zeros((10,10))):
    print('calculating the aucs for each grid point')
    auc_grid = np.zeros(y_true.shape[1:])

    days = [0,1,2,3]
    for d in days:#day
        count=0
        for i in range(y_true.shape[2]):#lat
            for j in range(y_true.shape[3]):#lon
                if count%100==0:
                    print(d,count,y_true.shape[2]*y_true.shape[3])
                y_true_pt = np.ravel(y_true[:,d,i,j])
                y_pred_pt = np.ravel(y_pred[:,d,i,j])
                precision, recall, thresholds = precision_recall_curve(y_true_pt,y_pred_pt)
                auc_grid[d,i,j] = auc(recall,precision)
                count+=1
                del y_true_pt, y_pred_pt, precision, recall, thresholds
    return auc_grid

def calc_podfar_gridpts(y_true=np.zeros((10,10)),y_pred=np.zeros((10,10)),threshold=.25):
    print('calculating the pods for each grid point')
    y_pred_thresholded = (y_pred>=threshold).astype(int)
    y_true = y_true.astype(int)
    precision_grid = np.zeros(y_true.shape[1:])
    recall_grid = np.zeros(y_true.shape[1:])
    for d in range(4):
        count=0
        for i in range(y_true.shape[2]):#lat
            for j in range(y_true.shape[3]):#lon
                if count%100==0:
                    print(d,count,y_true.shape[2]*y_true.shape[3])
                y_true_pt = np.ravel(y_true[:,d,i,j])
                y_pred_pt = np.ravel(y_pred_thresholded[:,d,i,j])
                recall_grid[d,i,j] = recall_score(y_true=y_true_pt, y_pred=y_pred_pt, labels=[0,1])
                precision_grid[d,i,j] = precision_score(y_true=y_true_pt, y_pred=y_pred_pt, labels=[0,1])
                count+=1
                del y_true_pt, y_pred_pt
    return recall_grid, precision_grid

def make_stat_fig(stat='recall'):
    print('making the %s figure'%stat)
    gfs_grid = pickle.load(open('gfs_grid.pkl','rb'))
    lat = gfs_grid['lat']
    lon = gfs_grid['lon']
    lon2d = gfs_grid['lon2d']
    lat2d = gfs_grid['lat2d']
    pe1 = [path_effects.withStroke(linewidth=1.5,foreground="k")]
    pe2 = [path_effects.withStroke(linewidth=1.5,foreground="w")]

    unet_stats = pickle.load(open('/scratch/bmac87/BC_reviewer_analysis/unet_%s_all.pkl'%stat,'rb'))
    lstm_stats = pickle.load(open('/scratch/bmac87/BC_reviewer_analysis/lstm_%s_all.pkl'%stat,'rb'))
    diff_stats = unet_stats-lstm_stats

    ttest_results = pickle.load(open('/scratch/bmac87/BC_reviewer_analysis/ttest.pkl','rb'))
    sig_mask = ttest_results['sig_mask']    

    edgecolor='black'
    fig, axes = plt.subplots(figsize=(32,16),
                            nrows=4,
                            ncols=3,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    subplot_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    label_count = 0
    for row in range(4):
        bounds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        cmap = plt.get_cmap('viridis')
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        vmin=0.0
        vmax=1.0
        sig_mask_np = sig_mask[row,:,:]
        for col in range(3):
            if col==0:
                data = unet_stats[row,:,:]
                fontcolor='white'
                edgecolor='white'
                axes[row,col].set_ylabel('Day %s'%(row+1))
            if col==1:
                data = lstm_stats[row,:,:]
                fontcolor='white'
                edgecolor='white'
            if col==2:
                data = diff_stats[row,:,:]
                if stat=='precision':
                    bounds = [-.5,-.45,-.40,-.35,-.30,-.25, -.2,-.15,-.1,-.05, 0, .05, .1, .15, .2, .25,.30,.35,.4,.45,.5]
                    vmin=-.5
                    vmax=.5
                if stat=='recall':
                    bounds = [-.5,-.45,-.40,-.35,-.30,-.25, -.2,-.15,-.1,-.05, 0, .05, .1, .15, .2, .25,.30,.35,.4,.45,.5]
                    vmin=-.5
                    vmax=.5
                if stat=='auc':
                    bounds = [-.35,-.30,-.25, -.2,-.15,-.1,-.05, 0, .05, .1, .15, .2, .25,.30,.35]
                    vmin=-.35
                    vmax=.35
                cmap = plt.get_cmap('coolwarm')
                norm = mcolors.BoundaryNorm(bounds, cmap.N)
                fontcolor='black'
                edgecolor='black'
            im = axes[row,col].pcolormesh(lon,lat,data,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree())
            axes[row,col].add_feature(cfeature.COASTLINE,edgecolor=edgecolor,linewidth=.25,transform=ccrs.PlateCarree())
            axes[row,col].add_feature(cfeature.STATES,edgecolor=edgecolor,linewidth=.25,transform=ccrs.PlateCarree())
            axes[row,col].text(236,24,subplot_labels[label_count],fontsize=24,color=fontcolor,weight='heavy',path_effects=pe1,transform=ccrs.PlateCarree())
            label_count+=1
            if col==2:
                axes[row,col].contourf(lon2d,lat2d,~sig_mask_np,hatches=['/',''],cmap='grey',alpha=.01)
                axes[row,col].contour(lon2d,lat2d,sig_mask_np,colors='black')
            divider = make_axes_locatable(axes[row,col])
            cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
            cb = fig.colorbar(im, cax=cax, orientation='vertical')
            cb.ax.tick_params(labelsize=14)
            del data
    axes[0,0].set_title('UNet',fontsize=24)
    axes[0,1].set_title('LSTM',fontsize=24)
    axes[0,2].set_title('Difference',fontsize=24)   
    plt.show()
    plt.savefig('./AIES_results_reviewer_edits/%s_map_all.png'%stat)
    plt.close()

if __name__=='__main__':

    make_stat_fig(stat='recall')
    make_stat_fig(stat='precision')
    make_stat_fig(stat='auc')