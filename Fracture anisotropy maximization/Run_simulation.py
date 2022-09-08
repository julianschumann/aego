#%%
import os
import numpy as np
from AEGO import AEGO 
from cost_function import Resnet_cost_function
from Encoder_and_Decoder import create_Encoder_and_Decoder
import tensorflow as tf
from mpi4py import MPI

#%% Setup MPI environment
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

#%% Setup requried classes (no randomness involved, so no information transfer required)
C = Resnet_cost_function(target_label = 656, use_softmax = True, gradient_loss_weight = 100.0)
Optimizer = AEGO(cost_function = C, max_num_samples = 25)
Networks = create_Encoder_and_Decoder()

#%% Prepare separate steps

generate_samples = True
determine_edim = True
train_new_networks = True
optimize_over_latent_space = True
post_process = True

#%% Generate training sampels
if generate_samples or not os.path.isfile('Results/Training_samples.npy'):
    np.random.seed(mpi_rank)
    X_rand_rank, C_rand_rank = Optimizer.generate_samples(num_samples = 50, LO_params = {'alpha': 0.01, 'beta_m': 0.9, 'beta_v': 0.999}, 
                                                          lam = 500, deflation = True, num_deflation_starts = 200,
                                                          use_MPI = True, mpi_comm = mpi_comm, mpi_rank = mpi_rank, mpi_size = mpi_size)
    
    mpi_comm.Barrier()
    if mpi_rank == 0:
        X_rand_receive = np.empty([mpi_size] + list(X_rand_rank))
        C_rand_receive = np.empty([mpi_size] + list(C_rand_rank))
    else:
        X_rand_receive = None
        C_rand_receive = None
        X_rand = None
        C_rand = None
        
    mpi_comm.Barrier()
    
    mpi_comm.Gather(X_rand_rank, X_rand_receive, root = 0)
    mpi_comm.Gather(C_rand_rank, C_rand_receive, root = 0)
    
    if mpi_rank == 0:
        X_shape = X_rand_receive.shape
        X_rand = np.zeros([X_shape[0] * X_shape[1]] + list(X_shape[2:]))
        
        C_shape = C_rand_receive.shape
        C_rand = np.zeros([C_shape[1], C_shape[0] * C_shape[2]])
        
        for i in range(mpi_size):
            X_rand[i * X_shape[1]:(i + 1) * X_shape[1],:] = X_rand_receive[i,:,:]
            C_rand[:, i * C_shape[2]:(i + 1) * C_shape[2]] = C_rand_receive[i,:,:]
    # Broadcast data to rest of processes
    X_rand = mpi_comm.bcast(X_rand, root = 0)
    C_rand = mpi_comm.bcast(C_rand, root = 0)
    
    if mpi_rank == 0:
        data_rand = np.array([X_rand.astype('float32'), C_rand, 0])
        np.save('Results/Training_samples.npy',data_rand)

else:
    data_rand = np.load('Results/Training_samples.npy', allow_pickle = True)
    [X_rand, C_rand, _] = data_rand 
    
    
#%% Determine intrinsic dimensionality (used on a gpu, so only rank == 0)
if determine_edim or not os.path.isfile('Results/Dimension.npy'):
    np.random.seed(0)
    tf.random.set_seed(0)
    
    Edim = [10,100]
    Loss = [] * len(Edim)
    EL = []
    DL = []
    
    for i,edim in enumerate(Edim):
        if EL == []:
            EL, DL = Networks.get_networks(edim)
            epochs = 50
        else:
            EL_new, DL_new = Networks.get_networks(edim)
            EL[-1] = EL_new[-1]
            DL[-1] = DL_new[-1]
            epochs = [0] * len(DL)
            epochs[-1] = 50 
            
        
        Encoder, Decoder, _ = Optimizer.pretrain_Decoder(EL, DL, X_rand, epochs = epochs, batch_size = 500)
    
        _, _, Loss[i] = Optimizer.train_Decoder(Encoder,  Decoder, X_rand, epochs = 100, batch_size = 500)
    i_chosen = np.argmin(Loss)
    #Try to replace with more
        
    edim = Edim[i_chosen]
    np.save('Results/Dimension.npy', np.array([edim, loss]))
else:
    edim = np.load('Results/Dimension.npy', allow_pickle = True)[0]
    
    
#%% Train decoder
if train_new_networks or not os.path.isfile('Results/Decoder.h5'):
    np.random.seed(0)
    tf.random.set_seed(0)
    EL, DL = Networks.get_networks(edim)     
    Encoder, Decoder, _ = Optimizer.pretrain_Decoder(EL, DL, X_rand, epochs = epochs, batch_size = 200)  
    Encoder, Decoder, _ = Optimizer.train_Decoder(Encoder, Decoder, X_rand, epochs = 200, batch_size = 200, 
                                                  use_Surr = True, C_train = C_rand[[-1],:].T, Surr_lam = 0.25, 
                                                  use_AAE = True, ad_prob = 0.25)
    
    Encoder.save('Results/Encoder.h5')
    Decoder.save('Results/Decoder.h5')
else:
    Decoder = tf.keras.models.load_model('Results/Decoder.h5')
    Optimizer.add_decoder(Decoder)

#%% Optimize over latent space

if optimize_over_latent_space or not os.path.isfile('Results/Optimization.npy'):
    np.random.seed(mpi_rank)
    Z_de_rank, X_de_rank, C_de_rank = Optimizer.DE(zmin = 0.0, zmax = 1.0, gamma = 100, G = 500, F = 0.6, chi_0 = 0.95, 
                                                   mu = 10, LO_params = {'alpha': 0.1, 'beta_m': 0.9, 'beta_v': 0.999}, 
                                                   use_MPI = False, mpi_comm = mpi_comm, mpi_rank = mpi_rank, mpi_size = mpi_size)
    mpi_comm.Barrier()
    if mpi_rank == 0:
        Z_de_receive = np.empty([mpi_size] + list(Z_de_rank.shape))
        X_de_receive = np.empty([mpi_size] + list(X_de_rank.shape))
        C_de_receive = np.empty([mpi_size] + list(C_de_rank.shape))
    else:
        Z_de_receive = None
        X_de_receive = None
        C_de_receive = None
        Z_de = None
        X_de = None
        C_de = None
        
    mpi_comm.Barrier()
    
    mpi_comm.Gather(Z_de_rank, Z_de_receive, root = 0)
    mpi_comm.Gather(X_de_rank, X_de_receive, root = 0)
    mpi_comm.Gather(C_de_rank, C_de_receive, root = 0)
    
    if mpi_rank == 0:
        Z_shape = Z_de_receive.shape
        Z_de = np.zeros([Z_shape[0] * Z_shape[1]] + list(Z_shape[2:]))
        
        X_shape = X_de_receive.shape
        X_de = np.zeros([X_shape[0] * X_shape[1]] + list(X_shape[2:]))
        
        C_shape = C_de_receive.shape
        C_de = np.zeros([C_shape[1], C_shape[0] * C_shape[2]])
        
        for i in range(mpi_size):
            Z_de[i * Z_shape[1]:(i + 1) * Z_shape[1],:] = Z_de_receive[i,:,:]
            X_de[i * X_shape[1]:(i + 1) * X_shape[1],:] = X_de_receive[i,:,:]
            C_de[:, i * C_shape[2]:(i + 1) * C_shape[2]] = C_de_receive[i,:,:]
    # Broadcast data to rest of processes
    Z_de = mpi_comm.bcast(Z_de, root = 0)
    X_de = mpi_comm.bcast(X_de, root = 0)
    C_de = mpi_comm.bcast(C_de, root = 0)
    
    if mpi_rank == 0:
        data_de = np.array([Z_de, X_de, C_de, 0])
        np.save('Results/Optimization.npy',data_de)

else:
    data_rand = np.load('Results/Optimization.npy', allow_pickle = True)
    [Z_de, X_de, C_de, _] = data_de 

#%% Post processing

if post_process or not os.path.isfile('Results/Post.npy'):
    np.random.seed(mpi_rank)
    
    
    X_de_rank = X_de[mpi_rank*int(len(X_de)/mpi_size):(mpi_rank + 1)*int(len(X_de)/mpi_size),:]
    
    X_post_rank, C_post_rank = Optimizer.Post_processing(X_de_rank, nu = 100, LO_params = {'alpha': 0.01, 'beta_m': 0.9, 'beta_v': 0.999},
                                                         use_MPI = False, mpi_comm = mpi_comm, mpi_rank = mpi_rank, mpi_size = mpi_size)
    
    mpi_comm.Barrier()
    if mpi_rank == 0:
        X_post_receive = np.empty([mpi_size] + list(X_post_rank.shape))
        C_post_receive = np.empty([mpi_size] + list(C_post_rank.shape))
    else:
        X_post_receive = None
        C_post_receive = None
        X_post = None
        C_post = None
        
    mpi_comm.Barrier()
    
    mpi_comm.Gather(X_post_rank, X_post_receive, root = 0)
    mpi_comm.Gather(C_post_rank, C_post_receive, root = 0)
    
    if mpi_rank == 0:        
        X_shape = X_post_receive.shape
        X_post = np.zeros([X_shape[0] * X_shape[1]] + list(X_shape[2:]))
        
        C_shape = C_post_receive.shape
        C_post = np.zeros([C_shape[1], C_shape[0] * C_shape[2]])
        
        for i in range(mpi_size):
            X_post[i * X_shape[1]:(i + 1) * X_shape[1],:] = X_post_receive[i,:,:]
            C_post[:, i * C_shape[2]:(i + 1) * C_shape[2]] = C_post_receive[i,:,:]
    # Broadcast data to rest of processes
    X_post = mpi_comm.bcast(X_post, root = 0)
    C_post = mpi_comm.bcast(C_post, root = 0)
    
    if mpi_rank == 0:
        data_post = np.array([X_post, C_post, 0])
        np.save('Results/Post.npy',data_post)

else:
    data_rand = np.load('Results/Post.npy', allow_pickle = True)
    [X_post, C_post, _] = data_post

