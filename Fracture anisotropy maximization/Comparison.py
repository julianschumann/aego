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
from cost_function_3 import TOP_cost_function
from cost_function import TO_cost_function
from mpi4py import MPI

#%% Solve some shit tensorflow bug
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


#%% Setup MPI environment
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

#%%
def run():
    #%% Setup requried classes (no randomness involved, so no information transfer required)
    C = TOP_cost_function(debug = False, elnums = 26, save_para_view = False)
    # C = TO_cost_function(debug = True)
    Optimizer = AEGO(cost_function = C)
    
    #%% Generate training sampels
    # np.random.seed(mpi_rank + 11)
    # X_rand_rank, X_rand_rank_init, C_rand_rank, G_rand_rank = Optimizer.generate_samples(num_samples = 18000, 
    #                                                                                      lam = 250, 
    #                                                                                      LO_params = {"ml": 0.001,
    #                                                                                                   "firstreinitialize": False,
    #                                                                                                   "vectorize": True}, 
    #                                                                                      deflation = False,
    #                                                                                      use_MPI = True, 
    #                                                                                      mpi_comm = mpi_comm, 
    #                                                                                      mpi_rank = mpi_rank, 
    #                                                                                      mpi_size = mpi_size)
    # #%%
    # mpi_comm.Barrier()
    # if mpi_rank == 0:
    #     X_rand_receive = np.empty([mpi_size] + list(X_rand_rank.shape))
    #     X_rand_init_receive = np.empty([mpi_size] + list(X_rand_rank_init.shape))
    #     C_rand_receive = np.empty([mpi_size] + list(C_rand_rank.shape))
    #     G_rand_receive = np.empty([mpi_size] + list(G_rand_rank.shape))
    # else:
    #     X_rand_receive = None
    #     X_rand_init_receive = None
    #     C_rand_receive = None
    #     G_rand_receive = None
    #     X_rand = None
    #     C_rand = None
    #     G_rand = None
        
    # mpi_comm.Barrier()
    
    # mpi_comm.Gather(X_rand_rank, X_rand_receive, root = 0)
    # mpi_comm.Gather(X_rand_rank_init, X_rand_init_receive, root = 0)
    # mpi_comm.Gather(C_rand_rank, C_rand_receive, root = 0)
    # mpi_comm.Gather(G_rand_rank, G_rand_receive, root = 0)
    
    # mpi_comm.Barrier()
    
    # if mpi_rank == 0:
    #     X_shape = X_rand_receive.shape
    #     X_rand = np.zeros([X_shape[0] * X_shape[1]] + list(X_shape[2:]))
        
    #     X_shape_init = X_rand_init_receive.shape
    #     X_rand_init = np.zeros([X_shape_init[0] * X_shape_init[1]] + list(X_shape_init[2:]))
        
    #     C_shape = C_rand_receive.shape
    #     C_rand = np.zeros([C_shape[1], C_shape[0] * C_shape[2]])
        
    #     G_shape = G_rand_receive.shape
    #     G_rand = np.zeros([G_shape[1], G_shape[0] * G_shape[2], G_shape[3]])
        
    #     for i in range(mpi_size):
    #         X_rand[i * X_shape[1]:(i + 1) * X_shape[1],:] = X_rand_receive[i,:,:]
    #         X_rand_init[i * X_shape_init[1]:(i + 1) * X_shape_init[1],:] = X_rand_init_receive[i,:,:]
    #         C_rand[:, i * C_shape[2]:(i + 1) * C_shape[2]] = C_rand_receive[i,:,:]
    #         G_rand[:, i * G_shape[2]:(i + 1) * G_shape[2], :] = G_rand_receive[i,:,:,:]
        
    #     data_rand = np.array([X_rand, X_rand_init, C_rand, G_rand, 0], object)
    #     np.save('Results/Comparison_LO.npy',data_rand)
    
    #%% Generate training sampels
    De_in = tf.keras.layers.Input(729)
    De_out = tf.keras.layers.Reshape((27,27))(De_in)

    Decoder = tf.keras.models.Model(De_in, De_out)
    Optimizer.add_decoder(Decoder)

    np.random.seed(mpi_rank)
    _, X_de_rank, C_de_rank, G_de_rank = Optimizer.DE(zmin = 0.0, 
                                                      zmax = 1.0, 
                                                      gamma = 300, 
                                                      G = 300, 
                                                      F = 0.6, 
                                                      chi_0 = 0.9, 
                                                      mu = 50, 
                                                      LO_params = {"ml": 0.001,
                                                                   "firstreinitialize": False,
                                                                   "vectorize": True}, 
                                                      use_MPI = True, 
                                                      mpi_comm = mpi_comm, 
                                                      mpi_rank = mpi_rank, 
                                                      mpi_size = mpi_size)

    mpi_comm.Barrier()
    if mpi_rank == 0:
        X_de_receive = np.empty([mpi_size] + list(X_de_rank.shape))
        C_de_receive = np.empty([mpi_size] + list(C_de_rank.shape))
        G_de_receive = np.empty([mpi_size] + list(G_de_rank.shape))
    else:
        X_de_receive = None
        C_de_receive = None
        G_de_receive = None
        X_de = None
        C_de = None
        G_de = None
        
    mpi_comm.Barrier()

    mpi_comm.Gather(X_de_rank, X_de_receive, root = 0)
    mpi_comm.Gather(C_de_rank, C_de_receive, root = 0)
    mpi_comm.Gather(G_de_rank, G_de_receive, root = 0)

    if mpi_rank == 0:
        X_shape = X_de_receive.shape
        X_de = np.zeros([X_shape[0] * X_shape[1]] + list(X_shape[2:]))
        
        C_shape = C_de_receive.shape
        C_de = np.zeros([C_shape[1], C_shape[0] * C_shape[2]])
        
        G_shape = G_de_receive.shape
        G_de = np.zeros([G_shape[1], G_shape[0] * G_shape[2], G_shape[3]])
        
        for i in range(mpi_size):
            X_de[i * X_shape[1]:(i + 1) * X_shape[1],:] = X_de_receive[i,:,:]
            C_de[:, i * C_shape[2]:(i + 1) * C_shape[2]] = C_de_receive[i,:,:]
            G_de[:, i * G_shape[2]:(i + 1) * G_shape[2], :] = G_de_receive[i,:,:,:]
    # Broadcast data to rest of processes
    X_de = mpi_comm.bcast(X_de, root = 0)
    C_de = mpi_comm.bcast(C_de, root = 0)
    G_de = mpi_comm.bcast(G_de, root = 0)

    if mpi_rank == 0:
        data_de = np.array([X_de, C_de, G_de, 0], object)
        np.save('Results/Comparison_DE.npy',data_de)
    
#%%
if __name__ == "__main__":
    run()