#%% Load packages
import builtins
try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile

import numpy as np
from cost_function_shear import TOP_cost_function
import matplotlib.pyplot as plt

#%% Load training data
file_name = '_shear'

C = TOP_cost_function(debug = True, elnums = 26, save_para_view = False, para_view_char = 0)
# Generate training sampels
data_rand = np.load('Training_samples' + file_name + '.npy', allow_pickle = True)
try:
    [X_rand, X_rand_init, C_rand, G_rand, _] = data_rand 
except: 
    [X_rand, C_rand, G_rand, _] = data_rand     
N_rand = np.arange(0, C_rand.size, C_rand.shape[1]) 
C_rand_min = np.min(C_rand, 1)
for i in range(len(C_rand_min) - 1):
    if C_rand_min[i+1] > C_rand_min[i]:
        C_rand_min[i+1] = C_rand_min[i]

#%% Load optimization results
# Determine intrinsic dimensionality (used on a gpu, so only rank == 0)
data_de = np.load('Optimization' + file_name + '.npy', allow_pickle = True)
[Z_de, X_de, C_de, G_de, _] = data_de 
mu = 100
N_de = np.arange(N_rand.max(), N_rand.max() + C_de.size * mu, C_de.shape[1] * mu)
C_de_min = np.min(C_de, 1)
for i in range(len(C_de_min) - 1):
    if C_de_min[i+1] > C_de_min[i]:
        C_de_min[i+1] = C_de_min[i]

# Post processing
data_post = np.load('Post' + file_name + '.npy', allow_pickle = True)
[X_post, C_post, G_post, _] = data_post
N_post = np.arange(N_de.max(), N_de.max() + C_post.size, C_de.shape[1])
C_post_min = np.min(C_post, 1)
for i in range(len(C_post_min) - 1):
    if C_post_min[i+1] > C_post_min[i]:
        C_post_min[i+1] = C_post_min[i]


#%% Load comparison
data_comp = np.load('Comparison_DE' + file_name + '.npy', allow_pickle = True)
[X_comp, C_comp, G_comp, _] = data_comp

N_comp = np.arange(0, C_comp.size * 50, C_comp.shape[1] * 50)
C_comp_min = np.min(C_comp, 1)
for i in range(len(C_comp_min) - 1):
    if C_comp_min[i+1] > C_comp_min[i]:
        C_comp_min[i+1] = C_comp_min[i]





#%% Plotting
plt.figure(figsize = (5,5))
# plt.plot(N_comp, C_comp_min, 'm')
plt.plot(N_rand, C_rand_min, 'r')
plt.plot(N_de, C_de_min, '--r')
plt.plot(N_post, C_post_min, ':r')
# plt.yscale('log')
plt.ylabel('min(c)')
plt.xlabel('n_e')
plt.tight_layout()
plt.show()


#%% Svae plot in txt file 
frand=open('f_Trand' + file_name + '.txt','w+')
frand.write('n c \n')
for k in range(len(C_rand_min)):
    frand.write('{:10.3e} {:10.3e} \n'.format(N_rand[k],C_rand_min[k]))
frand.close()

    
fae=open('f_Tae' + file_name + '.txt','w+')
fae.write('n c \n')
for k in range(len(C_de_min)):
    fae.write(' {:10.3e} {:10.3e} \n'.format(N_de[k],C_de_min[k]))
fae.close()


fpost=open('f_Tpost' + file_name + '.txt','w+')
fpost.write('n c \n')
for k in range(len(C_post_min)):
    fpost.write(' {:10.3e} {:10.3e} \n'.format(N_post[k],C_post_min[k]))
fpost.close()


fcomp=open('f_Tcomp' + file_name + '.txt','w+')
fcomp.write('n c \n')
for k in range(len(C_comp_min)):
    fcomp.write('{:10.3e} {:10.3e} \n'.format(N_comp[k],C_comp_min[k]))
fcomp.close()
#%% Save pictures
C.prepare_LO(num_samples = 1, ml = 0.001, firstreinitialize = False, vectorize = True)

I_post = np.where(C_post == np.min(C_post[(G_post[:,:,:2] < 0).all(-1)]))

C.save_para_view = True
X_post_best, C_post_best, _ = C.LO(X_de[I_post[1]], I_post[0][0])

C.create_picture(X_post_best, 'Design_post' + file_name)


C.prepare_LO(num_samples = 1, ml = 0.001, firstreinitialize = False, vectorize = True)

I_rand = np.where(C_rand == np.min(C_rand[(G_rand[:,:,:2] < 0).all(-1)]))

C.save_para_view = False
X_rand_best, C_rand_best, _ = C.LO(X_rand_init[I_rand[1]], I_rand[0][0])

C.create_picture(X_rand_best, 'Design_rand' + file_name)
