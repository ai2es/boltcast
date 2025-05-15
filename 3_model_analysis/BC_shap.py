import tensorflow as tf
import keras
import shap
import xarray as xr
from BC_analysis_data_loader import * 
tf.compat.v1.disable_v2_behavior()

def main():

    tf.config.set_visible_devices([], 'GPU')

    print('loading the tf model')
    model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/UNet_symmetric/all_rots/models/'
    model_fname = 'BC_UNet_rot_0_LR_0.000010000_deep_3_nconv_3_conv_size_4_stride_1_epochs_500__binary_batch_32_symmetric__SD_0.0_conv_relu__last_sigmoid__model.keras'
    model = tf.keras.models.load_model(model_dir+model_fname)

    print('loading the xr ds')
    ds_fname = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/tfds_binary_32batch/rot_0_test.nc'
    ds = xr.open_dataset(ds_fname,engine='netcdf4')
    print('normalizing the data')
    X_test_scaled = min_max_scale(ds)
    print('here are the input and output shapes')
    tfds_fname = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/tfds_binary_32batch/rot_0_test.tf'
    tfds = tf.data.Dataset.load(tfds_fname)
    print(tfds)

    print('constructing the explainer')
    
    # shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    expl = shap.DeepExplainer(model=(model.input, tf.keras.layers.Flatten()(model.output)),data=X_test_scaled[0:10,:,:,:,:])
    print('the explainer is constructed')
    print()
    print('calculating the shap values')
    shap_values = expl.shap_values(X_test_scaled[101:103,:,:,:,:],check_additivity=False)
    print('shap values calculated')


if __name__=='__main__':
    main()