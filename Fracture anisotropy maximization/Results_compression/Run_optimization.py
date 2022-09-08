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
import numpy as np
from AEGO import AEGO 
from cost_function_compression import TOP_cost_function
from mpi4py import MPI
#%% Setup MPI environment
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

#%% Setup requried classes (no randomness involved, so no information transfer required)
file_name = '_compression'


C = TOP_cost_function(debug = True, elnums = 26, save_para_view = False)
Optimizer = AEGO(cost_function = C)

Decoder = tf.keras.models.load_model('Decoder' + file_name + '.h5')
Optimizer.add_decoder(Decoder)

np.random.seed(mpi_rank)
Z_de_rank, X_de_rank, C_de_rank, G_de_rank = Optimizer.DE(zmin = 0.0, 
                                                          zmax = 1.0, 
                                                          gamma = 100, 
                                                          G = 250, 
                                                          F = 0.6, 
                                                          chi_0 = 0.9, 
                                                          mu = 100, 
                                                          LO_params = {"ml": 0.001,
                                                                       "firstreinitialize": False,
                                                                       "vectorize": True}, 
                                                          use_MPI = True, 
                                                          mpi_comm = mpi_comm, 
                                                          mpi_rank = mpi_rank, 
                                                          mpi_size = mpi_size)

mpi_comm.Barrier()
if mpi_rank == 0:
    Z_de_receive = np.empty([mpi_size] + list(Z_de_rank.shape))
    X_de_receive = np.empty([mpi_size] + list(X_de_rank.shape))
    C_de_receive = np.empty([mpi_size] + list(C_de_rank.shape))
    G_de_receive = np.empty([mpi_size] + list(G_de_rank.shape))
else:
    Z_de_receive = None
    X_de_receive = None
    C_de_receive = None
    G_de_receive = None
    Z_de = None
    X_de = None
    C_de = None
    G_de = None
    
mpi_comm.Barrier()

mpi_comm.Gather(Z_de_rank, Z_de_receive, root = 0)
mpi_comm.Gather(X_de_rank, X_de_receive, root = 0)
mpi_comm.Gather(C_de_rank, C_de_receive, root = 0)
mpi_comm.Gather(G_de_rank, G_de_receive, root = 0)

if mpi_rank == 0:
    Z_shape = Z_de_receive.shape
    Z_de = np.zeros([Z_shape[0] * Z_shape[1]] + list(Z_shape[2:]))
    
    X_shape = X_de_receive.shape
    X_de = np.zeros([X_shape[0] * X_shape[1]] + list(X_shape[2:]))
    
    C_shape = C_de_receive.shape
    C_de = np.zeros([C_shape[1], C_shape[0] * C_shape[2]])
    
    G_shape = G_de_receive.shape
    G_de = np.zeros([G_shape[1], G_shape[0] * G_shape[2], G_shape[3]])
    
    for i in range(mpi_size):
        Z_de[i * Z_shape[1]:(i + 1) * Z_shape[1],:] = Z_de_receive[i,:,:]
        X_de[i * X_shape[1]:(i + 1) * X_shape[1],:] = X_de_receive[i,:,:]
        C_de[:, i * C_shape[2]:(i + 1) * C_shape[2]] = C_de_receive[i,:,:]
        G_de[:, i * G_shape[2]:(i + 1) * G_shape[2], :] = G_de_receive[i,:,:,:]
# Broadcast data to rest of processes
Z_de = mpi_comm.bcast(Z_de, root = 0)
X_de = mpi_comm.bcast(X_de, root = 0)
C_de = mpi_comm.bcast(C_de, root = 0)
G_de = mpi_comm.bcast(G_de, root = 0)

if mpi_rank == 0:
    data_de = np.array([Z_de, X_de, C_de, G_de, 0], object)
    np.save('Optimization' + file_name + '.npy',data_de)

np.random.seed(mpi_rank)


X_de_rank = X_de[mpi_rank*int(len(X_de)/mpi_size):(mpi_rank + 1)*int(len(X_de)/mpi_size),:]

X_post_rank, C_post_rank, G_post_rank = Optimizer.Post_processing(X_de_rank, 
                                                                  nu = 500, 
                                                                  LO_params = {"ml": 0.001,
                                                                               "firstreinitialize": False,
                                                                               "vectorize": True}, 
                                                                  use_MPI = True, 
                                                                  mpi_comm = mpi_comm, 
                                                                  mpi_rank = mpi_rank, 
                                                                  mpi_size = mpi_size)

mpi_comm.Barrier()
if mpi_rank == 0:
    X_post_receive = np.empty([mpi_size] + list(X_post_rank.shape))
    C_post_receive = np.empty([mpi_size] + list(C_post_rank.shape))
    G_post_receive = np.empty([mpi_size] + list(G_post_rank.shape))
else:
    X_post_receive = None
    C_post_receive = None
    G_post_receive = None
    X_post = None
    C_post = None
    G_post = None
    
mpi_comm.Barrier()

mpi_comm.Gather(X_post_rank, X_post_receive, root = 0)
mpi_comm.Gather(C_post_rank, C_post_receive, root = 0)
mpi_comm.Gather(G_post_rank, G_post_receive, root = 0)

if mpi_rank == 0:        
    X_shape = X_post_receive.shape
    X_post = np.zeros([X_shape[0] * X_shape[1]] + list(X_shape[2:]))
    
    C_shape = C_post_receive.shape
    C_post = np.zeros([C_shape[1], C_shape[0] * C_shape[2]])
    
    G_shape = G_post_receive.shape
    G_post = np.zeros([G_shape[1], G_shape[0] * G_shape[2], G_shape[3]])
    
    for i in range(mpi_size):
        X_post[i * X_shape[1]:(i + 1) * X_shape[1],:] = X_post_receive[i,:,:]
        C_post[:, i * C_shape[2]:(i + 1) * C_shape[2]] = C_post_receive[i,:,:]
        G_post[:, i * G_shape[2]:(i + 1) * G_shape[2], :] = G_post_receive[i,:,:,:]
# Broadcast data to rest of processes
X_post = mpi_comm.bcast(X_post, root = 0)
C_post = mpi_comm.bcast(C_post, root = 0)
G_post = mpi_comm.bcast(G_post, root = 0)

if mpi_rank == 0:
    data_post = np.array([X_post, C_post, G_post, 0], object)
    np.save('Post' + file_name + '.npy',data_post)


