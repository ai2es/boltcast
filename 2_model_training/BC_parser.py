import argparse

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='BoltCast', fromfile_prefix_chars='@')

    # High-level info for WandB
    parser.add_argument('--project', type=str, default='BoltCast_postGE', help='WandB project name')

    # Project configuration
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', help='Perform the experiment even if the it was completed previously')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--load_data',action='store_true',default=False,help='Flag to load the data')
    parser.add_argument('--no-load_data',action='store_false',dest='load_data')
    parser.add_argument('--build_model',action='store_true',default=False,help='Flag to build the model')
    parser.add_argument('--no-build_model',action='store_false',dest='build_model')
    parser.add_argument('--rotation',type=int, default=0,help='The rotation for cross validation')
    parser.add_argument('--exp',type=int,default=1,help='The experiment number')
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')
    parser.add_argument('--image_size', nargs=4, type=int, default=[128,256,104], help="Size of input images (rows, cols, channels)")
    parser.add_argument('--results_path', type=str, default='/scratch/bmac87/BoltCast/results/', help='Results directory')
    parser.add_argument('--ckpt_path',type=str,default='/scratch/bmac87/BoltCast/model_checkpoints/',help='Model checkpoint directory')
    parser.add_argument('--render', action='store_true', default=False, help='Write model image')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')
    parser.add_argument('--model2train',type=str,default='UNet',help='The type of model to train')
    
    # Specific experiment configuration
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files")
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=0.00001, help="Learning rate")
    parser.add_argument('--loss',type=str,default='binary-cross-entropy',help='The loss function of the optimizer')
    parser.add_argument('--metrics',type=str,default='accuracy',help='The metric to monitor')
    
    parser.add_argument('--spatial_dropout',type=float,default=0.0,help='Amount of spatial dropout')
    parser.add_argument('--batch_normalization', action='store_true', help='Turn on batch normalization')
    parser.add_argument('--no-batch_normalization',action='store_false',dest='batch_normalization')
    parser.add_argument('--L2_reg',type=float,default=0.0,help='L2 Regularization rate')
    parser.add_argument('--early_stopping',action='store_true',default=False,help='Use Early Stopping')
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")
    
    parser.add_argument('--batch', type=int, default=16, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=3, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4, help="Number of threads to use during batch construction")
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--no-repeat',action='store_false',dest='repeat')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Number of training batches per epoch (must use --repeat if you are using this)")
    
    # U-Net
    parser.add_argument('--deep',type=int,default=2,help='How deep to build the model')
    parser.add_argument('--n_conv_per_step',type=int,default=2,help='The number of convolutions per deep step')
    parser.add_argument('--conv_size', type=int, default=3, help='Convolution filter size per layer')
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--pool', type=int, default=2, help='Max pooling size (1=None)')
    parser.add_argument('--stride',type=int,default=1,help='Stride pixels')
    parser.add_argument('--padding', type=str, default='same', help='Padding type for convolutional layers')
    parser.add_argument('--activation_conv', type=str, default='elu', help='Activation function for convolutional layers')
    parser.add_argument('--activation_last',type=str,default='sigmoid',help='Last activation function')
    parser.add_argument('--skip',action='store_true',default=False,help='Build skip connections in the UNet')
    parser.add_argument('--no-skip',action='store_false',dest='skip',help='Do no use skip connections in the UNet')

    #LSTM
    parser.add_argument('--lstm_conv_deep',type=int,default=1,help='The number of convolutional layers deep')
    parser.add_argument('--lstm_conv_size',type=int,default=1,help='The kernel size')
    parser.add_argument('--lstm_pool',type=int,default=2,help='Pooling in the LSTM model')
    parser.add_argument('--lstm_padding',type=str,default='same',help='The padding type for the model')
    parser.add_argument('--lstm_conv_activation',type=str,default='relu',help='The activation function')
    parser.add_argument('--lstm_stride',type=int,default=1)
    parser.add_argument('--lstm_activation_last',type=str,default='sigmoid',help='The last activation function')
    parser.add_argument('--lstm_deep',type=int,default=1,help='The number of LSTM layers')
    parser.add_argument('--return_sequences',action='store_true',default=False,help='Whether or not to return the sequences from the LSTM layer')
    parser.add_argument('--no-return_sequences',action='store_false',dest='return_sequences')
    parser.add_argument('--return_state',action='store_true',default=False,help='Whether or not to return the state of the LSTM layer')
    parser.add_argument('--no-return_state',action='store_false',dest='return_state')

    return parser

