import os
# import tensorflow as tf
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd

def main():
    print('in BChyper.py main')
    activations = ['elu','relu']
    L2s = [0.0, 1e-5, 1e-4, 1e-3]
    drops = [0.0,0.05,0.10]
    convs_2d = [2,3,4]
    
    conv_list = []
    drop_list = []
    L2_list = []
    act_list = []

    for act in activations:
        for L2 in L2s:
            for drop in drops:
                for conv in convs_2d:
                    act_list.append(act)
                    L2_list.append(L2)
                    drop_list.append(drop)
                    conv_list.append(conv)
    
    hyper_dict = {'activation':act_list,
                    'L2':L2_list,
                    'dropout':drop_list,
                    'conv_size':conv_list}
    df = pd.DataFrame(data=hyper_dict)
    print(df)
    df.to_pickle(open('hyper_dict.pkl','rb'))

if __name__=='__main__':
    main()