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
from cost_function_expansion import TOP_cost_function
from mpi4py import MPI

#%% Solve some shit tensorflow bug
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


#%% Setup MPI environment
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

file_name = '_expansion'
#%%
def run():
    #%% Setup requried classes (no randomness involved, so no information transfer required)
    C = TOP_cost_function(debug = False, elnums = 26, save_para_view = False)
    # C = TO_cost_function(debug = True)
    Optimizer = AEGO(cost_function = C)
    
    #%% Generate training sampels
    np.random.seed(mpi_rank + 11)
    X_rand_rank, X_rand_rank_init, C_rand_rank, G_rand_rank = Optimizer.generate_samples(num_samples = 6000, 
                                                                                         lam = 250, 
                                                                                         LO_params = {"ml": 0.001,
                                                                                                      "firstreinitialize": False,
                                                                                                      "vectorize": True}, 
                                                                                         deflation = False,
                                                                                         use_MPI = True, 
                                                                                         mpi_comm = mpi_comm, 
                                                                                         mpi_rank = mpi_rank, 
                                                                                         mpi_size = mpi_size)
    #%%
    mpi_comm.Barrier()
    if mpi_rank == 0:
        X_rand_receive = np.empty([mpi_size] + list(X_rand_rank.shape))
        X_rand_init_receive = np.empty([mpi_size] + list(X_rand_rank_init.shape))
        C_rand_receive = np.empty([mpi_size] + list(C_rand_rank.shape))
        G_rand_receive = np.empty([mpi_size] + list(G_rand_rank.shape))
    else:
        X_rand_receive = None
        X_rand_init_receive = None
        C_rand_receive = None
        G_rand_receive = None
        X_rand = None
        C_rand = None
        G_rand = None
        
    mpi_comm.Barrier()
    
    mpi_comm.Gather(X_rand_rank, X_rand_receive, root = 0)
    mpi_comm.Gather(X_rand_rank_init, X_rand_init_receive, root = 0)
    mpi_comm.Gather(C_rand_rank, C_rand_receive, root = 0)
    mpi_comm.Gather(G_rand_rank, G_rand_receive, root = 0)
    
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        X_shape = X_rand_receive.shape
        X_rand = np.zeros([X_shape[0] * X_shape[1]] + list(X_shape[2:]))
        
        X_shape_init = X_rand_init_receive.shape
        X_rand_init = np.zeros([X_shape_init[0] * X_shape_init[1]] + list(X_shape_init[2:]))
        
        C_shape = C_rand_receive.shape
        C_rand = np.zeros([C_shape[1], C_shape[0] * C_shape[2]])
        
        G_shape = G_rand_receive.shape
        G_rand = np.zeros([G_shape[1], G_shape[0] * G_shape[2], G_shape[3]])
        
        for i in range(mpi_size):
            X_rand[i * X_shape[1]:(i + 1) * X_shape[1],:] = X_rand_receive[i,:,:]
            X_rand_init[i * X_shape_init[1]:(i + 1) * X_shape_init[1],:] = X_rand_init_receive[i,:,:]
            C_rand[:, i * C_shape[2]:(i + 1) * C_shape[2]] = C_rand_receive[i,:,:]
            G_rand[:, i * G_shape[2]:(i + 1) * G_shape[2], :] = G_rand_receive[i,:,:,:]
        
        data_rand = np.array([X_rand, X_rand_init, C_rand, G_rand, 0], object)
        np.save('Training_samples' + file_name + '.npy',data_rand)
    
    return X_rand, X_rand_init, C_rand, G_rand
    
#%%
if __name__ == "__main__":
    X_rand, X_rand_init, C_rand, G_rand = run()