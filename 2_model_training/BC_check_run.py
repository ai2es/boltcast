import os
import pickle

model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/LSTM/models/'
exp_id = 0
exp_dict = {'-1':{'rot':0,'lstm_deep':0,'conv_deep':0}}
for conv_deep in [0,1,2]:
    for lstm_deep in [1,2,3]:
        for rot in [0,1,2,3,4]:
            print(exp_id)
            fname = 'BC_LSTM_rot_%s_conv_4_conv_deep_%s_lstm_deep_%s_no_drop_no_shuffle_model.keras'%(rot,conv_deep,lstm_deep)
            temp_dict = {'rot':rot,'lstm_deep':lstm_deep,'conv_deep':conv_deep}
            exp_dict.update({str(exp_id):temp_dict})
            exp_id+=1
del exp_dict['-1']
pickle.dump(exp_dict,open('./lstm_exp_dict.pkl','wb'))

