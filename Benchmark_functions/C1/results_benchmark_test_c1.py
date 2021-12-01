import numpy as np


data=np.load('Results/Benchmark_data_edim=10.npy',allow_pickle=True)
[C_min_rand,C_min_ae,C_min_post,eval_rand,eval_ae,eval_post,_]=data
data=np.load('Results/Benchmark_data_edim=5.npy',allow_pickle=True)
[C_min_rand_red,C_min_ae_red,C_min_post_red,eval_rand_red,eval_ae_red,eval_post_red,_]=data

data=np.load('Results/Benchmark_comp_data.npy',allow_pickle=True)
[C_min_ae_de,C_min_post_de,eval_ae_de,eval_post_de,_]=data



fpost=open('f_post.txt','w+')
fpost.write('n c \n')
for k in range(len(C_min_post)):
    fpost.write('{:10.3e} {:10.3e} \n'.format(eval_post[k],max(min(C_min_post[k],np.min(C_min_post[:k+1])),10e-6)))
fpost.close()

fae=open('f_ae.txt','w+')
fae.write('n c \n')
for k in range(len(C_min_ae)):
    fae.write('{:10.3e} {:10.3e} \n'.format(eval_ae[k],max(min(C_min_ae[k],np.min(C_min_ae[:k+1])),10e-6)))
fae.close()

frand=open('f_rand.txt','w+')
frand.write('n c \n')
for k in range(len(C_min_rand)):
    frand.write('{:10.4e} {:10.4e} \n'.format(eval_rand[k],max(min(C_min_rand[k],np.min(C_min_rand[:k+1])),10e-6)))
frand.close()


fpost_red=open('f_post_red.txt','w+')
fpost_red.write('n c \n')
for k in range(len(C_min_ae_red)):
    fpost_red.write('{:10.3e} {:10.3e} \n'.format(eval_post_red[k],max(min(C_min_ae_red[k],np.min(C_min_ae_red[:k+1])),10e-6)))
fpost_red.close()

fae_red=open('f_ae_red.txt','w+')
fae_red.write('n c \n')
for k in range(len(C_min_ae_red)):
    fae_red.write('{:10.3e} {:10.3e} \n'.format(eval_ae_red[k],max(min(C_min_ae_red[k],np.min(C_min_ae_red[:k+1])),10e-6)))
fae_red.close()


fpost_de=open('f_post_de.txt','w+')
fpost_de.write('n c \n')
for k in range(len(C_min_post_de)):
    fpost_de.write('{:10.3e} {:10.3e} \n'.format(eval_post_de[k],max(min(C_min_post_de[k],np.min(C_min_post_de[:k+1])),10e-6)))
fpost_de.close()

fae_de=open('f_ae_de.txt','w+')
fae_de.write('n c \n')
for k in range(len(C_min_ae_de)):
    fae_de.write('{:10.3e} {:10.3e} \n'.format(eval_ae_de[k],max(min(C_min_ae_de[k],np.min(C_min_ae_de[:k+1])),10e-6)))
fae_de.close()