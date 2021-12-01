import numpy as np
from mpi4py import MPI
from keras.models import Model
from keras.layers import Input
from create_neural_network_models import create_model_max, create_discriminator
from autoencoder_training import train_autoencoder, train_AAE_surr
































comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#possibly also doing pretraining or not


################################################################
####                 Load data                              ####
################################################################
filename_simp='Sample_data/X_rand.npy'
filename_C='Sample_data/C_rand.npy'

if rank==0: 
    X_simp=np.load(filename_simp) 
    
    C=np.load(filename_C)
    
    Index=np.arange(C.size)
    np.random.shuffle(Index)
    
    X_simp=X_simp[Index[:4000],:,:]
    C=C[Index[:4000]]
    
    volfrac=np.mean(X_simp)
else:
    X_simp=None
    C=None
    Index=None
    volfrac=None

comm.Barrier()

X_simp=comm.bcast(X_simp,root=0)
C=comm.bcast(C,root=0)
volfrac=comm.bcast(volfrac,root=0)

(_,nely,nelx)=X_simp.shape

n=len(C)
perrank=int(n/size)

X_simp_rank=X_simp[rank*perrank:(rank+1)*perrank,:,:]
C_rank=C[rank*perrank:(rank+1)*perrank,np.newaxis]


W_simp_rank=np.ones(X_simp_rank.shape)
W_simp_rank=W_simp_rank+1-2*np.abs(X_simp_rank-0.5)

perrank_train=int(0.8*perrank)
perrank_val=perrank-perrank_train

X_simp_rank_train=X_simp_rank[:perrank_train,:,:]
X_simp_rank_val=X_simp_rank[perrank_train:,:,:]

W_simp_rank_train=W_simp_rank[:perrank_train,:,:]
W_simp_rank_val=W_simp_rank[perrank_train:,:,:]

C_rank_train=C_rank[:perrank_train,:]


edim=100
pre=1
dis=1
surr=1

n_epochs=250
n_epochs_pre=25
typ='max'
               
if rank==0:
    print('edim={}, pre={}, AAE={}, surr={}, typ='.format(edim,pre,dis,surr)+typ)
comm.Barrier()
################################################################################################
####                                   Training                                             ####
################################################################################################
Eo, Ei, Di, Do=create_model_max(nelx,nely,edim)
X_rank_train=X_simp_rank_train
X_rank_val=X_simp_rank_val
W_rank_train=W_simp_rank_train
W_rank_val=W_simp_rank_val


Design_space=Input((nely,nelx))
Latent_space=Input((edim,))
Middle_layer=Input((Ei.input.shape[1],))
AEo=Model(Design_space, Do(Eo(Design_space)))
AEi=Model(Middle_layer, Di(Ei(Middle_layer)))
# Train the outer layers
AEo_weights=train_autoencoder(AEo,X_rank_train,np.ones(X_rank_train.shape), comm, rank, size, perrank_train,n_epochs_pre)
AEo.set_weights(AEo_weights) 
Xm_rank=Eo.predict(X_rank_train) 
# Train the inner layers
AEi_weights=train_autoencoder(AEi,Xm_rank,np.ones(Xm_rank.shape), comm, rank, size, perrank_train,n_epochs_pre)
AEi.set_weights(AEi_weights)
# Put together networks
Encoder=Model(Design_space,Ei(Eo(Design_space)))
Decoder=Model(Latent_space,Do(Di(Latent_space)))
Autoencoder=Model(Design_space, Decoder(Encoder(Design_space)))     
# Train main networks        
Discriminator=create_discriminator(edim)
Surrogate=create_discriminator(edim)  
AE_weights=train_AAE_surr(Autoencoder,Encoder,Discriminator,Surrogate,X_rank_train, W_rank_train,C_rank_train,  comm,rank, size, perrank_train,n_epochs)                   
Autoencoder.set_weights(AE_weights)  
#Save results
if rank==0:
    Decoder.save('Sample_data/Decoder_random.h5')

            
                        
                    


