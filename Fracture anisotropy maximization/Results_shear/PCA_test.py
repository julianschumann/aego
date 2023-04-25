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
from sklearn.decomposition import PCA   
import matplotlib.pyplot as plt
from AEGO import AEGO 
from cost_function_shear import TOP_cost_function
import os

file_name = '_shear'

if (not os.path.isfile('PCA' + file_name + '.npy')) and (not os.path.isfile('PCA_rand' + file_name + '.npy')):
    C = TOP_cost_function(debug = True, elnums = 26, save_para_view = False)
    Optimizer = AEGO(cost_function = C)
    
    data_rand = np.load('Training_samples' + file_name + '.npy', allow_pickle = True)
    [X_rand,X_comp,_, _,_] = data_rand 
    
    print("Max value X_rand: {}".format(np.max(np.abs(X_rand))))
    print("Max value X_comp: {}".format(np.max(np.abs(X_comp))))
    
    X_rand = ((X_rand - Optimizer.x_min_value) / Optimizer.x_interval).reshape(len(X_rand),-1)
    
    X_comp = ((X_comp - Optimizer.x_min_value) / Optimizer.x_interval).reshape(len(X_rand),-1)
    
    Edim = [1,2,5,10,20,35,50,75,100,150,200,300,400,500]
    loss = np.zeros(len(Edim))
    loss_comp = np.zeros(len(Edim))
    
    
    for i,edim in enumerate(Edim):
        pca = PCA(edim)
        pca.fit(X_rand)
        loss[i] = np.mean((X_rand - pca.inverse_transform(pca.transform(X_rand)))**2)
        if edim == Edim[-1]:
            Part = np.cumsum(pca.explained_variance_ratio_)
        
        pca.fit(X_comp)
        loss_comp[i] = np.mean((X_comp - pca.inverse_transform(pca.transform(X_comp)))**2)
        if edim == Edim[-1]:
            Part_comp = np.cumsum(pca.explained_variance_ratio_)
    
    data = np.array([Edim, loss, Part, 0], object)
    data_comp = np.array([Edim, loss_comp, Part_comp, 0], object)
    
    np.save('PCA' + file_name + '.npy', data)
    np.save('PCA_rand' + file_name + '.npy', data_comp)
else:
    Edim, loss, Part, _ = np.load('PCA' + file_name + '.npy', allow_pickle = True)
    Edim, loss_comp, Part_comp, _ = np.load('PCA_rand' + file_name + '.npy', allow_pickle = True)

data_ae = np.load('Dimension' + file_name + '.npy', allow_pickle = True)
Edim_ae, loss_ae = data_ae

loss_max = max(loss.max(), loss_comp.max(), loss_ae.max())
loss_min = min(loss.min(), loss_comp.min(), loss_ae.min())

fig, ax1 = plt.subplots()
color = '#ff0000'
ax1.set_xlabel('m')
ax1.set_ylabel('PCA reconstruction loss', color=color)
ax1.plot(Edim, loss, color = color, label = 'locally optimized - PCA')
ax1.plot(Edim_ae, loss_ae, color = '#ff4040', label = 'locally optimized - AE')
ax1.plot(Edim,loss_comp, color = '#ff8080', label = 'random baseline')
ax1.set_xlim([0,Edim[-1]])
ax1.set_yscale('log')
ax1.set_ylim([10 ** np.floor(np.log(loss_min)/np.log(10)), 10 ** np.ceil(np.log(loss_max)/np.log(10))])
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = '#0000ff'
ax2.set_ylabel('explained variance ratio', color=color)  # we already handled the x-label with ax1
ax2.plot(np.arange(Edim[-1] + 1), np.concatenate(([0], Part)), color = color, label = 'locally optimized - PCA')
ax2.plot(np.arange(Edim[-1] + 1), np.concatenate(([0], Part_comp)), color = '#8080ff', label = 'random baseline')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0,1])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.legend(loc='center left', bbox_to_anchor=(0.6, 0.5))
plt.show()



#%% Save results to text for paper figure 
fpca=open('f_PCA_loss' + file_name + '.txt','w+')
fpca.write('m L Lr \n')
for k in range(len(Edim)):
    fpca.write('{:10.3e} {:10.3e} {:10.3e}\n'.format(Edim[k], loss[k], loss_comp[k]))
fpca.close()

fae=open('f_AE_loss' + file_name + '.txt','w+')
fae.write('m L \n')
for k in range(len(Edim_ae)):
    fae.write('{:10.3e} {:10.3e}\n'.format(Edim_ae[k], loss_ae[k]))
fae.close()

fvar=open('f_PCA_var' + file_name + '.txt','w+')
fvar.write('m V Vr \n')
fvar.write('{:10.3e} {:10.3e} {:10.3e}\n'.format(0, 0, 0))
for k in range(Edim[-1]):
    fvar.write('{:10.3e} {:10.3e} {:10.3e}\n'.format(k + 1, Part[k], Part_comp[k]))
fvar.close()
