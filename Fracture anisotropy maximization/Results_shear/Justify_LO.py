#%% Avoid error with @profile declarator
import builtins
try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile
    
#%%
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
import numpy as np
from AEGO import AEGO 
from cost_function_shear import TOP_cost_function
from create_neural_network_models import Autoencoder
import os
from mpi4py import MPI
#%% Setup MPI environment
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

file_name = '_shear'

# Setup requried classes (no randomness involved, so no information transfer required)
C = TOP_cost_function(debug = True, elnums = 26, save_para_view = False)
Optimizer = AEGO(cost_function = C)
# Load training data
data_rand = np.load('Training_samples' + file_name + '.npy', allow_pickle = True)
[XL, XF, C_rand, _, _] = data_rand 
CL = C_rand[-1]
CF = C_rand[0]

#%% Train AE on random samples
if not os.path.isfile('Decoder_rand' + file_name + '.h5'):
    C = TOP_cost_function(debug = True, elnums = 26, save_para_view = False)
    Optimizer = AEGO(cost_function = C)
    # Load training data
    
    tf.random.set_seed(0)
    np.random.seed(0)
    
    Networks = Autoencoder(XF.shape[1], XF.shape[2])
    
    edim = 50
    EL, DL = Networks.get_networks(edim)     
    Encoder_rand, Decoder_rand, _ = Optimizer.pretrain_Decoder(EL, DL, XF, epochs = 100, batch_size = 100)  
    Encoder_rand, Decoder_rand, _ = Optimizer.train_Decoder(Encoder_rand, Decoder_rand, XF, epochs = 200, batch_size = 100, 
                                                            use_Surr = True, C_train = C_rand[[-1],:].T, Surr_lam = 0.25, 
                                                            use_AAE = True, ad_prob = 0.3)
    
    Encoder_rand.save('Encoder_rand' + file_name + '.h5')
    Decoder_rand.save('Decoder_rand' + file_name + '.h5')
else:
    Decoder_rand = tf.keras.models.load_model('Decoder_rand' + file_name + '.h5')
    
 
#%% Evaluate latent_space samples
# Numbers test:
num = len(XL)

if not os.path.isfile('C_AE' + file_name + '.npy'):
    # Run over random AE with mu = 0:
    Optimizer.add_decoder(Decoder_rand)
    np.random.seed(mpi_rank)
    Z_ae_rand_0_rank, X_ae_rand_0_rank, C_ae_rand_0_rank, _ = Optimizer.DE(zmin = 0.0, zmax = 1.0, 
                                                                           gamma = num, G = 0, mu = 0, 
                                                                           LO_params = {"ml": 0.001,
                                                                                        "firstreinitialize": False,
                                                                                        "vectorize": True}, 
                                                                           use_MPI = True, mpi_comm = mpi_comm, 
                                                                           mpi_rank = mpi_rank, mpi_size = mpi_size)
    mpi_comm.Barrier()
    
    # Run over random AE with mu = 100:
    np.random.seed(mpi_rank)
    Z_ae_rand_100_rank, X_ae_rand_100_rank, C_ae_rand_100_rank, _ = Optimizer.DE(zmin = 0.0, zmax = 1.0, 
                                                                                 gamma = num, G = 0, mu = 100, 
                                                                                 LO_params = {"ml": 0.001,
                                                                                              "firstreinitialize": False,
                                                                                              "vectorize": True}, 
                                                                                 use_MPI = True, mpi_comm = mpi_comm, 
                                                                                 mpi_rank = mpi_rank, mpi_size = mpi_size)
    mpi_comm.Barrier()    
    
    # run over real AE
    Decoder = tf.keras.models.load_model('Decoder' + file_name + '.h5')
    Optimizer.add_decoder(Decoder)
    np.random.seed(mpi_rank)
    Z_ae_0_rank, X_ae_0_rank, C_ae_0_rank, _ = Optimizer.DE(zmin = 0.0, zmax = 1.0, 
                                                            gamma = num, G = 0, mu = 0, 
                                                            LO_params = {"ml": 0.001,
                                                                         "firstreinitialize": False,
                                                                         "vectorize": True}, 
                                                            use_MPI = True, mpi_comm = mpi_comm, 
                                                            mpi_rank = mpi_rank, mpi_size = mpi_size)
    mpi_comm.Barrier()
    
    # Run over random AE with mu = 100:
    np.random.seed(mpi_rank)
    Z_ae_100_rank, X_ae_100_rank, C_ae_100_rank, _ = Optimizer.DE(zmin = 0.0, zmax = 1.0, 
                                                                  gamma = num, G = 0, mu = 100, 
                                                                  LO_params = {"ml": 0.001,
                                                                               "firstreinitialize": False,
                                                                               "vectorize": True}, 
                                                                  use_MPI = True, mpi_comm = mpi_comm, 
                                                                  mpi_rank = mpi_rank, mpi_size = mpi_size)
    mpi_comm.Barrier()    
    
    #%% Save those results
    Z_de_rank = np.stack([Z_ae_rand_0_rank, Z_ae_rand_100_rank, Z_ae_0_rank, Z_ae_100_rank], axis = 1)
    X_de_rank = np.stack([X_ae_rand_0_rank, X_ae_rand_100_rank, X_ae_0_rank, X_ae_100_rank], axis = 1)
    C_de_rank = np.stack([C_ae_rand_0_rank, C_ae_rand_100_rank, C_ae_0_rank, C_ae_100_rank], axis = -1)
    
    
    
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
        C_de = np.zeros([C_shape[1], C_shape[0] * C_shape[2], C_shape[-1]])
        
        for i in range(mpi_size):
            Z_de[i * Z_shape[1]:(i + 1) * Z_shape[1]] = Z_de_receive[i]
            X_de[i * X_shape[1]:(i + 1) * X_shape[1]] = X_de_receive[i]
            C_de[:, i * C_shape[2]:(i + 1) * C_shape[2]] = C_de_receive[i]
    
    # Broadcast data to rest of processes
    Z_de = mpi_comm.bcast(Z_de, root = 0)
    X_de = mpi_comm.bcast(X_de, root = 0)
    C_de = mpi_comm.bcast(C_de, root = 0)
    
    if mpi_rank == 0:
        data_de = np.array([Z_de, X_de, C_de, 0], object)
        np.save('C_AE' + file_name + '.npy', data_de)
else:
    [Z_de, X_de, C_de, _] = np.load('C_AE' + file_name + '.npy', allow_pickle = True)
    
C_de = C_de.transpose(2,0,1)

#%% Get probabilities

if not os.path.isfile('Prob' + file_name + '.npy'):
    def pdf(C,K,c):
        C = np.sort(C)
        Cp = np.pad(C, K+1, 'edge')
        Cp[:K] = 2 * C[0] - C[-1] - 1
        Cp[-K:] = 2 * C[-1] - C[0] + 1
        p=np.zeros(len(c))
        for i in range(len(c)):
            ci = c[i]
            k = np.searchsorted(C,ci)
            Ci = Cp[k+1 : k+1 + 2*K]
            Di = np.abs(Ci-ci)
            R = np.median(np.concatenate((Di,[0])))
            p[i]=K/R
        p=p/(2*len(C))
        p[1:-1] = 0.25 * (2 * p[1:-1] + p[:-2] + p[2:])
        return p
    
    Type = np.array(['X_0', 'X_250', 'AE_0_(X_0)', 'AE_100_(X_0)', 'AE_0_(X_250)', 'AE_100_(X_250)']) 
    C = np.stack([CF, CL, C_de[0,0], C_de[1,0], C_de[2,0], C_de[3,0]])
    P = []
    
    c_min = C.min()
    c_max = C.max()
    c_int = c_max - c_min
    
    c_range = np.linspace(c_min - 0.25 * c_int, c_max + 0.25 * c_int, 501)
    k = 60
    for c in C:
        p = pdf(c, k, c_range)
        P.append(p)
    
    P = np.array(P)
    
    data_prob = [Type, C, P, 0]
    np.save('Prob' + file_name + '.npy', data_prob)
else:
    [Type, C, P, _] = np.load('Prob' + file_name + '.npy', allow_pickle = True)


f=open('probability' + file_name + '.txt','w+')
f.write('c ' + ' '.join(Type) + '\n')
for i in range(len(c)):
    f.write('{:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e}\n'.format(c[i], *P[:,i]))
    
f.close()