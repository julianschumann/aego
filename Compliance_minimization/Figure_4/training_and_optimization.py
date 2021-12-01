import numpy as np
from mpi4py import MPI
from create_neural_network_models import create_model_max, create_model_sdf, create_discriminator
from keras.models import Model
from keras.layers import Input
from SIMP import TO_SIMP, make_Conn_matrix
import time
from autoencoder_training import train_autoencoder, train_AAE, train_AAE_surr, train_autoencoder_surr
















def get_void(nely,nelx):
    v=np.zeros((nely,nelx))
    R=min(nely,nelx)/15
    loc=np.array([[1/3, 1/4], [2/3, 1/4],[ 1/3, 1/2], [2/3, 1/2], [1/3 , 3/4], [2/3, 3/4]])
    loc=loc*np.array([[nely,nelx]])
    for i in range(nely):
        for j in range(nelx):
            v[i,j]=R-np.min(np.sqrt(np.sum((loc-np.array([[i+1,j+1]]))**2,1)));
    v=v>0
    return v

def evaluate_design(Z,Decoder,volfrac,Iar,cMat,void,opt_it,typ):
    beta=0.05
    epsilon_1=1
    epsilon_2=0.25
    nelx=90
    nely=45
    penal=3
    E0=1
    nu=0.3
    max_move=0.25
    
    X=Decoder.predict(Z)
    if typ=='sdf':
        X=np.clip(X+0.5,0,1)
           
    (n,nely,nelx)=X.shape
    avoid=np.zeros((1,nely,nelx))
    C=np.zeros(n)
    X_out=np.zeros(X.shape)
    for i in range(n):
        X_out[i,:,:], _    = TO_SIMP(X[i,:,:]    , volfrac, penal, beta, epsilon_1, max_move, E0, nu, Iar, cMat, False, void, avoid, 0, opt_it)
        ## Enfroce sparse solution
        X_out[i,:,:], C[i] = TO_SIMP(X_out[i,:,:], volfrac, penal, beta, epsilon_2, max_move, E0, nu, Iar, cMat, True , void, avoid, 0, 10)
    return X_out,C



## Implent multiprocessing, with size processors, where rank is the one currently executing this file
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Define possible variations
## Two different latent space dimensionalities
encoding_dim=np.array([25,100])
## Two different types of encoding a design ('max'=density field, 'sdf'=signed distance field)
NN_typ=['max','sdf']
## Two different numbers of optimizations steps mu in the latent space cost function
Opt_it=np.array([1,25])
## The number of the optimization run to be performed (needed so that one can save data from multiple different runs)
example_cases=4

## Number of epocs used for autoencoder training (and if neccessary for autoencoder pretraining)
n_epochs=250
n_epochs_pre=50


for example_case in range(example_cases):
    if rank==0: 
        ## Load the data of the training samples
        filename_simp='Sample_data/X_simp.npy'
        filename_sdf='Sample_data/X_sdf.npy'
        filename_C='Sample_data/C.npy'
    
        X_simp=np.load(filename_simp)
        X_simp=X_simp.reshape((X_simp.shape[0]*X_simp.shape[1],X_simp.shape[2],X_simp.shape[3]))
        
        X_sdf=np.load(filename_sdf)
        X_sdf=X_simp.reshape((X_sdf.shape[0]*X_sdf.shape[1],X_sdf.shape[2],X_sdf.shape[3]))
        
        C=np.load(filename_C)
        C=C.reshape((C.size,1))
        
        ## Randomize order of training samples to get random training and validation sets
        Index=np.arange(C.size)
        np.random.shuffle(Index)
        
        X_simp=X_simp[Index,:,:]
        X_sdf=X_sdf[Index,:,:]
        C=C[Index,:]
        
        ## Estimate mass constraint from training samples
        volfrac=np.mean(X_simp)
    else:
        X_simp=None
        X_sdf=None
        C=None
        Index=None
        volfrac=None
    
    comm.Barrier()
    
    ## Broadcast the traing samples ofer all ranks
    X_simp=comm.bcast(X_simp,root=0)
    X_sdf=comm.bcast(X_sdf,root=0)
    C=comm.bcast(C,root=0)
    volfrac=comm.bcast(volfrac,root=0)
    ## Choose the training samples for each rank
    n=len(C)
    perrank=int(n/size)
    X_simp_rank=X_simp[rank*perrank:(rank+1)*perrank,:,:]
    X_sdf_rank=X_sdf[rank*perrank:(rank+1)*perrank,:,:]
    C_rank=C[rank*perrank:(rank+1)*perrank,:]
    
    ## Get parameters important for the cost function
    (_,nely,nelx)=X_simp.shape
    void=get_void(nely, nelx)
    Iar,cMat=make_Conn_matrix(nelx,nely)
    
    ## Generate the weights for the autoencoder training
    W_simp_rank=np.ones(X_simp_rank.shape)
    W_simp_rank=W_simp_rank+1-2*np.abs(X_simp_rank-0.5)
    
    W_sdf_rank=np.ones(X_sdf_rank.shape)
    W_sdf_rank[np.abs(X_sdf_rank)>1]=0.5
    W_sdf_rank[np.abs(X_sdf_rank)>2]=0.25
    W_sdf_rank[np.abs(X_sdf_rank)>5]=0.1
    W_sdf_rank[np.abs(X_sdf_rank)>10]=0.025
    W_sdf_rank[np.abs(X_sdf_rank)>20]=0.01
    
    
    ## Divide samples in traing and validation set
    perrank_train=int(0.8*perrank)
    perrank_val=perrank-perrank_train
    
    X_simp_rank_train=X_simp_rank[:perrank_train,:,:]
    X_simp_rank_val=X_simp_rank[perrank_train:,:,:]
    
    W_simp_rank_train=W_simp_rank[:perrank_train,:,:]
    
    X_sdf_rank_train=X_sdf_rank[:perrank_train,:,:]
    X_sdf_rank_val=X_sdf_rank[perrank_train:,:,:]
    
    W_sdf_rank_train=W_sdf_rank[:perrank_train,:,:]
    
    C_rank_train=C_rank[:perrank_train,:]
    
    
    ## Start the whole traing and optimization process (I think a very small amount of time could be changed by changing the order of variations, but such benefits would be negliable)
    
    ## Vary the latent space dimensionality
    for edim in encoding_dim:
        ## Vary the use of pretraining for encoder and decoder
        for pre in range(2):
            ## Vary the use of a discriminator network
            for dis in range(2):
                ## Vary the use of a surrogate model network
                for surr in range(2):
                    ## Vary the type of encoding of designs
                    for typ in NN_typ: 
                        if rank==0:
                            print('edim={}, pre={}, AAE={}, surr={}, typ='.format(edim,pre,dis,surr)+typ)
                        comm.Barrier()
                        start_time=time.time()
                        ## Train the neural networks:
                        
                        ## Implement the type of encoding of the network and generate encoder and decoder networks
                        if typ=='max':
                            Eo, Ei, Di, Do=create_model_max(nelx,nely,edim)
                            X_rank_train=X_simp_rank_train
                            X_rank_val=X_simp_rank_val
                            W_rank_train=W_simp_rank_train
                        else:
                            Eo, Ei, Di, Do=create_model_sdf(nelx,nely,edim)      
                            X_rank_train=X_sdf_rank_train
                            X_rank_val=X_sdf_rank_val
                            W_rank_train=W_sdf_rank_train
                        
                        ## Prepare traing
                        Design_space=Input((nely,nelx))
                        Latent_space=Input((edim,))
                        ## Pretraining, when used, is implemented here
                        if pre==1:
                            Middle_layer=Input((Ei.input.shape[1],))
                            AEo=Model(Design_space, Do(Eo(Design_space)))
                            AEi=Model(Middle_layer, Di(Ei(Middle_layer)))
                            ## The outer, convolutional part of the network is trained
                            AEo_weights=train_autoencoder(AEo,X_rank_train,np.ones(X_rank_train.shape),comm, rank, size, perrank_train,n_epochs_pre)
                            AEo.set_weights(AEo_weights) 
                            Xm_rank=Eo.predict(X_rank_train) 
                            ## The inner, dense part of the autoencoder is trained
                            AEi_weights=train_autoencoder(AEi,Xm_rank,np.ones(Xm_rank.shape), comm,rank, size, perrank_train,n_epochs_pre)
                            AEi.set_weights(AEi_weights)
                        ## The encoder and decoder networks, as well as the autoencoder network is build by assembling the parts
                        Encoder=Model(Design_space,Ei(Eo(Design_space)))
                        Decoder=Model(Latent_space,Do(Di(Latent_space)))
                        Autoencoder=Model(Design_space, Decoder(Encoder(Design_space)))     
                        
                        
                        ## Surrogate model and discriminator networks are prepared (although they are not necessarily used every time)   
                        Discriminator=create_discriminator(edim)  
                        Surrogate=create_discriminator(edim)
                        
                        ## The whole network is trained, with different types being used
                        if dis==1:
                            if surr==1:  
                                AE_weights=train_AAE_surr(Autoencoder,Encoder,Discriminator,Surrogate,X_rank_train, W_rank_train,C_rank_train, comm,rank, size, perrank_train,n_epochs)                        
                            else:
                                AE_weights=train_AAE(Autoencoder,Encoder,Discriminator,X_rank_train, W_rank_train, comm,rank, size, perrank_train,n_epochs)                        
                        else:
                            if surr==1:  
                                AE_weights=train_autoencoder_surr(Autoencoder,Encoder,Surrogate,X_rank_train, W_rank_train, C_rank_train, rank, size, perrank_train,n_epochs)              
                            else:
                                AE_weights=train_autoencoder(Autoencoder,X_rank_train,W_rank_train, comm,rank, size, perrank_train,n_epochs)                               
                        Autoencoder.set_weights(AE_weights)  
                        
                        ## the resulting networks are saved (and yes, savinf the decoder is not really necessary, but is done nonetheless)
                        if rank==0:
                            Decoder.save('Sample_data/edim={}_pre={}_AAE={}_surr={}_'.format(edim,pre,dis,surr)+typ+'_case={}_Decoder.h5'.format(example_case))  
                            Autoencoder.save('Sample_data/edim={}_pre={}_AAE={}_surr={}_'.format(edim,pre,dis,surr)+typ+'_case={}_Autoencoder.h5'.format(example_case))
                        
                        ## Training and validation errors are calculated
                        if typ=='sdf':
                            X_sdf_pred_rank_train=Autoencoder.predict(X_sdf_rank_train)
                            X_sdf_pred_rank_val=Autoencoder.predict(X_sdf_rank_val)
                            X_simp_pred_rank_train=np.clip(X_sdf_pred_rank_train+0.5,0,1)
                            X_simp_pred_rank_val=np.clip(X_sdf_pred_rank_val+0.5,0,1)
                        else:
                            X_simp_pred_rank_train=Autoencoder.predict(X_simp_rank_train)
                            X_simp_pred_rank_val=Autoencoder.predict(X_simp_rank_val)
                        train_losses_rank=np.zeros(1)
                        val_losses_rank=np.zeros(1)
                        train_losses_rank[0]=np.mean((X_simp_pred_rank_train-X_simp_rank_train)**2)     
                        val_losses_rank[0]=np.mean((X_simp_pred_rank_val-X_simp_rank_val)**2)                       
                        comm.Barrier()
                        stop_time=time.time()
                        if rank==0:
                            train_losses_rec=np.zeros((size,1))
                            val_losses_rec=np.zeros((size,1))
                        else:
                            train_losses_rec=None 
                            val_losses_rec=None
                        comm.Gather(train_losses_rank,train_losses_rec,root=0)
                        comm.Gather(val_losses_rank,val_losses_rec,root=0)
                        
                        ## Training and validation errors are saved
                        if rank==0:
                            train_losses_rec=np.mean(train_losses_rec)
                            val_losses_rec=np.mean(val_losses_rec)
                            np.save('Sample_data/edim={}_pre={}_AAE={}_surr={}_'.format(edim,pre,dis,surr)+typ+'_case={}_training_losses.npy'.format(example_case),train_losses_rec)
                            np.save('Sample_data/edim={}_pre={}_AAE={}_surr={}_'.format(edim,pre,dis,surr)+typ+'_case={}_validation_losses.npy'.format(example_case),val_losses_rec)
                        if rank==0:
                            print('    training time: {}s'.format(int(10*(stop_time-start_time))/10))
                        
                        
                        ## Vary the number of local optimization steps in the cost function
                        for opt_it in Opt_it: 
                            ## Optimization over latent space is now performed using differential evolution (Storn and Price, 1997)
                            comm.Barrier()
                            start_time=time.time()
                            
                            ## Set differential evolution hyperparameters are set
                            num_pop_perrank=int(np.ceil(2.5*edim/size))
                            num_pop=size*num_pop_perrank
                            prob_change=0.9
                            multiplyer=0.6
                            
                            ## Initial generation is generated and evaluated
                            Z_rank=np.random.rand(num_pop_perrank,edim)
                            _,F_rank=evaluate_design(Z_rank,Decoder,volfrac,Iar,cMat,void,opt_it,typ)
                            comm.Barrier()
                            ## Initial generation is shared over all ranks
                            if rank==0:
                                Z_rec=np.empty((size,num_pop_perrank,edim))
                                F_rec=np.empty((size,num_pop_perrank))
                            else:
                                Z_rec=None
                                F_rec=None
                            comm.Gather(Z_rank,Z_rec,root=0)
                            comm.Gather(F_rank,F_rec,root=0)
                            if rank==0:
                                Z=Z_rec.reshape((num_pop,edim))
                                F=F_rec.reshape(num_pop)
                            else:
                                Z=None
                                F=None                                
                            Z=comm.bcast(Z,root=0)
                            F=comm.bcast(F,root=0)
                            loop=0
                            ## Generations are evolved
                            while loop<500:
                                Z_rank=Z[rank*num_pop_perrank:(rank+1)*num_pop_perrank,:]
                                F_rank=F[rank*num_pop_perrank:(rank+1)*num_pop_perrank]
                                
                                ## Reproduction between differnt individuals from the population is perforemd
                                test_case=np.floor(np.random.rand(num_pop_perrank,3)*(num_pop-1e-7)).astype('int')
                                Za_rank=np.copy(Z[test_case[:,0],:])
                                Zb_rank=np.copy(Z[test_case[:,1],:])
                                Zc_rank=np.copy(Z[test_case[:,2],:])
                                Zcom_rank=Za_rank+multiplyer*(Zb_rank-Zc_rank)
                                
                                ## Crossover between child and parent is performed
                                prob=np.random.rand(num_pop_perrank,edim)
                                Zcom_rank[prob>prob_change]=np.copy(Z_rank[prob>prob_change])
                                
                                ## Boundaries of design are enforced
                                Zcom_rank[Zcom_rank<0]=0
                                Zcom_rank[Zcom_rank>1]=1 
                                
                                ## Selection between child (has to be evaluated first) and parent is performed
                                _,F_compare=evaluate_design(Zcom_rank,Decoder,volfrac,Iar,cMat,void,opt_it,typ)
                                F_rank=np.minimum(F_rank,F_compare)
                                Z_rank[F_compare<=F_rank,:]=Zcom_rank[F_compare<=F_rank,:]
                                
                                ## New population is shared between all ranks
                                if rank==0:
                                    Z_rec=np.empty((size,num_pop_perrank,edim))
                                    F_rec=np.empty((size,num_pop_perrank))
                                else:
                                    Z_rec=None
                                    F_rec=None
                                comm.Barrier()
                                comm.Gather(Z_rank,Z_rec,root=0)
                                comm.Gather(F_rank,F_rec,root=0)
                                if rank==0:
                                    Z=Z_rec.reshape((num_pop,edim))
                                    F=F_rec.reshape(num_pop)
                                Z=comm.bcast(Z,root=0)
                                F=comm.bcast(F,root=0)
                                loop=loop+1
                            
                            ## Postprocessing is performed (all final designs from differential evolution are used, as due to parallel processing, the incresed time is negliable)
                            Z_rank=Z[rank*num_pop_perrank:(rank+1)*num_pop_perrank,:]
                            X_simp_final_rank,F_final_rank=evaluate_design(Z_rank,Decoder,volfrac,Iar,cMat,void,300,typ)
                            
                            ## Postprocessed designs are shared between ranks
                            if rank==0:
                                X_final_rec=np.empty((size,num_pop_perrank,nely,nelx))
                                F_final_rec=np.empty((size,num_pop_perrank))
                            else:
                                X_final_rec=None
                                F_final_rec=None
                            comm.Barrier()
                            comm.Gather(X_simp_final_rank,X_final_rec,root=0)
                            comm.Gather(F_final_rank,F_final_rec,root=0)
                            
                            ## Postprocessed designs are saved
                            if rank==0:
                                X_final=X_final_rec.reshape((num_pop,nely,nelx))
                                F_final=F_final_rec.reshape(num_pop)
                                data=[X_final.astype('float32'),F_final.astype('float32'),0]
                                np.save('Sample_data/edim={}_pre={}_AAE={}_surr={}_'.format(edim,pre,dis,surr)+typ+'_opt={}_case={}_Results.npy'.format(opt_it,example_case),np.array(data))
                            comm.Barrier()
                            stop_time=time.time()
                            if rank==0:
                                print('    optimizing opt={} time: {}s'.format(opt_it,int(10*(stop_time-start_time))/10))
                        # Now try just random local approach to see if global optimization is really necessary
                        comm.Barrier()
                
                        
                    


