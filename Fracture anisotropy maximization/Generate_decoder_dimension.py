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
    
    Networks = Autoencoder(X_rand.shape[1], X_rand.shape[2])
    
    tf.random.set_seed(0)
    np.random.seed(0)
    
    Edim = [1, 2, 5, 10, 20, 50, 100]
    Loss = [np.nan] * len(Edim)
    EL = []
    DL = []
    
    epochs = 100
    for i,edim in enumerate(Edim):
        for j in range(3):
            if j == 0:
                EL, DL = Networks.get_networks(edim)
        
                Encoder, Decoder, _ = Optimizer.pretrain_Decoder(EL, DL, X_rand, epochs = epochs, batch_size = 100)
            
                _, _, Loss[i] = Optimizer.train_Decoder(Encoder,  Decoder, X_rand, epochs = 100, batch_size = 100)
            else:
                EL, DL = Networks.get_networks(edim)
        
                Encoder, Decoder, _ = Optimizer.pretrain_Decoder(EL, DL, X_rand, epochs = epochs, batch_size = 100)
            
                _, _, lossij = Optimizer.train_Decoder(Encoder,  Decoder, X_rand, epochs = 100, batch_size = 100)
                if Loss[i] > lossij:
                    Loss[i] = lossij
         
    np.save('Results_4/Dimension.npy', np.array([Edim, Loss], object))
if __name__ == "__main__":
    run()