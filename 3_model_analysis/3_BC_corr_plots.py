import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle

def corr_plot(rot=4,
                ds_type='train'):
    
    load_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/binary/'
    fload = 'rot_%s_%s.tf'%(rot,ds_type)

    tfds = tf.data.Dataset.load(load_dir+fload)
    print(tfds)

    c=0
    for in_for,out_for in tfds:
        print(c)
        if c==0:
            inputs=in_for
            print('inputs.shape',inputs.shape)
        else:
            data = in_for
            # print('data.shape',data.shape)
            inputs = np.concatenate([inputs,data],axis=0)
            # print('inputs.shape',inputs.shape)
        c+=1

    ravel_np = np.zeros([inputs.shape[0]*inputs.shape[1]*inputs.shape[2]*inputs.shape[3],9])
    
    for i in range(9):
        ravel_np[:,i] = np.ravel(inputs[:,:,:,:,i])
    # print('ravel_np.shape:', ravel_np.shape)

    corr_coef_np = np.corrcoef(ravel_np,rowvar=False)
    del tfds, inputs, ravel_np, data 
    return corr_coef_np

def make_corr_plot(corr_coef_np,ds_type):
    vars = ['CAPE',
                'Lifted Index',
                'Reflectivity',
                'Precipitation Rate',
                'Vertical Velocity',
                'Ice Mixing Ratio',
                'Snow Mixing Ratio',
                'Graupel Mixing Ratio',
                'Rain Mixing Ratio']

    fig,ax = plt.subplots(1,1,figsize=(30,30))
    ax.matshow(corr_coef_np,cmap='coolwarm')
    ax.set_xticks(range(9),vars,fontsize=24,rotation=45)
    ax.set_yticks(range(9),vars,fontsize=24,rotation=0)
    ax.xaxis.tick_bottom()
    for i in range(9):
        for j in range(9):
            corr = corr_coef_np[i,j]
            r_str = f"{corr:.2f}"
            ax.text(i,j,r_str,fontsize=24)

    save_dir = '/home/bmac87/BoltCast/3_model_analysis/correlation_matrix_plots/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    fsave = 'corr_plot_%s.png'%(ds_type)
    plt.savefig(save_dir+fsave)
    fsave = 'corr_plot_%s.pdf'%(ds_type)
    plt.savefig(save_dir+fsave)
    plt.close()

if __name__=="__main__":
    train_4 = corr_plot(rot=4,ds_type='train')
    test_4 = corr_plot(rot=4,ds_type='test')

    train_3 = corr_plot(rot=3,ds_type='train')
    test_3 = corr_plot(rot=3,ds_type='test')

    train_2 = corr_plot(rot=2,ds_type='train')
    test_2 = corr_plot(rot=2,ds_type='test')

    train_1 = corr_plot(rot=1,ds_type='train')
    test_1 = corr_plot(rot=1,ds_type='test')

    train_0 = corr_plot(rot=0,ds_type='train')
    test_0 = corr_plot(rot=0,ds_type='test')

    train_3d = np.stack([train_4,train_3,train_2,train_1,train_0],axis=2)
    print(train_3d.shape)
    train_corr = np.mean(train_3d,axis=2)
    print(train_corr.shape)
    pickle.dump(train_3d,open('train_corr_3d.pkl','wb'))

    test_3d = np.stack([test_4,test_3,test_2,test_1,test_0],axis=2)
    print(test_3d.shape)
    test_corr = np.mean(test_3d,axis=2)
    print(test_corr.shape)
    pickle.dump(test_3d,open('test_corr_3d.pkl','wb'))

    make_corr_plot(corr_coef_np=test_corr,ds_type='test')
    make_corr_plot(corr_coef_np=train_corr,ds_type='train')

