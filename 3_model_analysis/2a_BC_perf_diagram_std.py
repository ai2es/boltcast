import numpy as np
import os
import pickle

def calc_std():
    print('calculating auc stds')
    lstm_dir = './LSTM_daily_dict/'
    unet_dir = './UNet_daily_dict/'
    
    conv_deep = 0
    lstm_deep = 1

    rots = [0,1,2,3,4]
    days = ['day_1','day_2','day_3','day_4']
    aucs = np.zeros((2,4,5))
    for rot in rots:
        lstm_file = 'daily_dict_rot_%s_conv_deep_%s_lstm_deep_%s.pkl'%(rot,conv_deep,lstm_deep)
        unet_file = 'daily_dict_rot_%s_unet.pkl'%(rot)

        unet_stats = pickle.load(open(unet_dir+unet_file,'rb'))
        lstm_stats = pickle.load(open(lstm_dir+lstm_file,'rb'))
        
        for d,day in enumerate(days):
            print(d,day)
            aucs[0,d,rot] = unet_stats[day]['auc']
            aucs[1,d,rot] = lstm_stats[day]['auc']
    print('unet stds')
    print(np.std(aucs[0,:,:],axis=1))

    print('lstm stds')
    print(np.std(aucs[1,:,:],axis=1))
def main():
    calc_std()

if __name__=='__main__':
    main()