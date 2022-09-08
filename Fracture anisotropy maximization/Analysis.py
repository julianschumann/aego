#%% Load packages
import builtins
try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile

import numpy as np
from cost_function_4 import TOP_cost_function
import matplotlib.pyplot as plt
import tikzplotlib as plt2tikz

#%% Load training data

Path_add_on = '_4'

C = TOP_cost_function(debug = True, elnums = 26, save_para_view = False, para_view_char = 0)
# Generate training sampels
data_rand = np.load('Results' + Path_add_on + '/Training_samples.npy', allow_pickle = True)
try:
    [X_rand, X_rand_init, C_rand, G_rand, _] = data_rand 
except: 
    [X_rand, C_rand, G_rand, _] = data_rand     
N_rand = np.arange(0, C_rand.size, C_rand.shape[1]) 
C_rand_min = np.min(C_rand, 1)
for i in range(len(C_rand_min) - 1):
    if C_rand_min[i+1] > C_rand_min[i]:
        C_rand_min[i+1] = C_rand_min[i]
#%% Get comperative result
lam = len(C_rand) - 1

X_orig_in = np.zeros([1] + list(X_rand.shape[1:]))
radius = 0.105
centers = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]

for i in range(X_orig_in.shape[1]):
    for j in range(X_orig_in.shape[2]):
        x = i/(X_orig_in.shape[1] - 1)
        y = j/(X_orig_in.shape[2] - 1)
        d = np.zeros(len(centers), float)
        for n, center in enumerate(centers):
            xc = center[0]
            yc = center[1]
            d[n] = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - radius
        X_orig_in[0, i, j] = - d.min()

X_min, X_max = C.get_boundaries()
X_orig_in = np.maximum(np.minimum(X_orig_in, X_max[np.newaxis,:,:]), X_min[np.newaxis,:,:])

C.prepare_LO(num_samples = 1, ml = 0.001, firstreinitialize = False, vectorize = True)
X_orig, C_orig, G_orig = C.LO(X_orig_in, num_steps = lam)





# C.create_picture(X_orig_in)
# C.create_picture(X_orig)
        

# plt.figure()
# for i in range(len(C_orig)):
#     ii = i / (len(C_orig) - 1)
#     if ii > 0 and ii < 1:
#         plt.scatter(G_orig[[i],2], C_orig[[i]], c = (ii, 1 - ii, (1 - ii) ** 2), marker = 'x', s = 10)
#     else:
#         plt.scatter(G_orig[[i],2], C_orig[[i]], c = (ii, 1 - ii, (1 - ii) ** 2), marker = 'x', s = 10, label = 'iteration: {}'.format(i))
# plt.scatter(G_orig[i_unsafe,2], C_orig[i_unsafe], marker = 'x', color = '#000000', s = 20)
# plt.ylabel('w * J1 - (1 - w) * J2')
# plt.xlabel('J2 / J1')
# plt.legend()
# plt.tight_layout()
# plt.show()

# Volume is either 0.501 * (1 + G_orig[:,0]) = 0.499 * (1 - G_orig[:,1])
# J1 = 2 * C_orig / (1 - G_orig[:,2])
# J2 = 2 * C_orig * G_orig[:,2] / (1 - G_orig[:,2])
#%% Load optimization results
# Determine intrinsic dimensionality (used on a gpu, so only rank == 0)
data_de = np.load('Results' + Path_add_on + '/Optimization.npy', allow_pickle = True)
[Z_de, X_de, C_de, G_de, _] = data_de 
mu = 100
N_de = np.arange(N_rand.max(), N_rand.max() + C_de.size * mu, C_de.shape[1] * mu)
C_de_min = np.min(C_de, 1)
for i in range(len(C_de_min) - 1):
    if C_de_min[i+1] > C_de_min[i]:
        C_de_min[i+1] = C_de_min[i]

# Post processing
data_post = np.load('Results' + Path_add_on + '/Post.npy', allow_pickle = True)
[X_post, C_post, G_post, _] = data_post
N_post = np.arange(N_de.max(), N_de.max() + C_post.size, C_de.shape[1])
C_post_min = np.min(C_post, 1)
for i in range(len(C_post_min) - 1):
    if C_post_min[i+1] > C_post_min[i]:
        C_post_min[i+1] = C_post_min[i]


#%% Load comparison
data_comp = np.load('Results' + Path_add_on + '/Comparison_DE.npy', allow_pickle = True)
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
frand=open('Results' + Path_add_on + '/f_Trand' + Path_add_on + '.txt','w+')
frand.write('n c \n')
for k in range(len(C_rand_min)):
    frand.write('{:10.3e} {:10.3e} \n'.format(N_rand[k],C_rand_min[k]))
frand.close()

    
fae=open('Results' + Path_add_on + '/f_Tae' + Path_add_on + '.txt','w+')
fae.write('n c \n')
for k in range(len(C_de_min)):
    fae.write(' {:10.3e} {:10.3e} \n'.format(N_de[k],C_de_min[k]))
fae.close()


fpost=open('Results' + Path_add_on + '/f_Tpost' + Path_add_on + '.txt','w+')
fpost.write('n c \n')
for k in range(len(C_post_min)):
    fpost.write(' {:10.3e} {:10.3e} \n'.format(N_post[k],C_post_min[k]))
fpost.close()


fcomp=open('Results' + Path_add_on + '/f_Tcomp' + Path_add_on + '.txt','w+')
fcomp.write('n c \n')
for k in range(len(C_comp_min)):
    fcomp.write('{:10.3e} {:10.3e} \n'.format(N_comp[k],C_comp_min[k]))
fcomp.close()
#%% Save pictures



C.prepare_LO(num_samples = 1, ml = 0.001, firstreinitialize = False, vectorize = True)

I_rand = np.where(C_rand == np.min(C_rand[(G_rand[:,:,:2] < 0).all(-1)]))

X_rand_best, C_rand_best, _ = C.LO(X_rand_init[I_rand[1]], I_rand[0][0])

C.create_picture(X_rand_best, 'Results' + Path_add_on + '/Design_rand' + Path_add_on)



i_de = np.argmin(C_de[-1])

X_de_best = X_de[i_de]

C.create_picture(X_de_best, 'Results' + Path_add_on + '/Design_de' + Path_add_on)



C.prepare_LO(num_samples = 1, ml = 0.001, firstreinitialize = False, vectorize = True)

I_post = np.where(C_post == np.min(C_post[(G_post[:,:,:2] < 0).all(-1)]))

X_post_best, C_post_best, _ = C.LO(X_de[I_post[1]], I_post[0][0])

C.create_picture(X_post_best, 'Results' + Path_add_on + '/Design_post' + Path_add_on)

