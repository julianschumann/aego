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
from cost_function_4 import TOP_cost_function
from cost_function import TO_cost_function
from create_neural_network_models import Autoencoder

def run():    
    # Setup requried classes (no randomness involved, so no information transfer required)
    C = TOP_cost_function(debug = True, elnums = 26, save_para_view = False)
    Optimizer = AEGO(cost_function = C)
    # Load training data
    data_rand = np.load('Results_4/Training_samples.npy', allow_pickle = True)
    [X_rand, _, C_rand, _, _] = data_rand 
        
    tf.random.set_seed(0)
    np.random.seed(0)
    
    Networks = Autoencoder(X_rand.shape[1], X_rand.shape[2])
    
    edim = 50
    EL, DL = Networks.get_networks(edim)     
    Encoder, Decoder, _ = Optimizer.pretrain_Decoder(EL, DL, X_rand, epochs = 100, batch_size = 100)  
    Encoder, Decoder, _ = Optimizer.train_Decoder(Encoder, Decoder, X_rand, epochs = 200, batch_size = 100, 
                                                  use_Surr = True, C_train = C_rand[[-1],:].T, Surr_lam = 0.25, 
                                                  use_AAE = True, ad_prob = 0.3)

    Encoder.save('Results_4/Encoder.h5')
    Decoder.save('Results_4/Decoder.h5')

if __name__ == "__main__":
    run()