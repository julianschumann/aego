import numpy as np
import matplotlib.pyplot as plt



Edim=[10]     
plt.figure(figsize=(6,6))
for edim in Edim:    
    data=np.load('Results/Benchmark_data_edim={}.npy'.format(edim),allow_pickle=True)            
    [C_rand,C_ae,C_post,eval_rand,eval_ae,eval_post,_]=data
    plt.plot(np.concatenate((eval_rand,eval_ae,eval_post),0),np.concatenate((C_rand,C_ae,C_post),0),label='m={}'.format(edim))
plt.ylabel('c')
plt.xlabel('n_f')
plt.yscale('log')
plt.show()