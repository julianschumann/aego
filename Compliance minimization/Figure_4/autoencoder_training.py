import numpy as np
from mpi4py import MPI
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
import tensorflow as tf
import keras.backend as K 

## These functions are used to train specific neural netoworks
## See Appendices A.6, A.7, and A.9 of my master thesis 

## Only the traing of the adversarial autoencoder with surrogate model will recieve code comments,
## as the other traiing function are basically the same only with cutout parts


 
def train_AAE_surr(AE,Encoder,Discriminator,surrogate,X_rank,W_rank,C_rank,comm,rank,size,perrank,n_epochs):
    '''
    Trains an adversarial autoencoder with an additional surrogate model network and discriminator.

    Parameters
    ----------
    AE : Model
        Autoencoder constisting out of encoder and decoder.
    Encoder : Model
        Encoder part of the autoencoder AE.
    Discriminator : Model
        The disriminator network.
    surrogate : Model
        The surrogate model network.
    X_rank : perrank*nely*nelx float
        perrank training samples for the autoencoder.
    W_rank : perrank*nely*nelx float
        Weights assigned to each training sample.
    C_rank : perrank float
        The compliance of each training sample.
    comm : MPI_WORLD
        Envrionment to communicate with different processors.
    rank : int
        The processor used here.
    size : int
        Total number of processors.
    perrank : int
        Number of training samples in each rank.
    n_epochs : int
        Number of training epochs.

    Returns
    -------
    AE_weights : Model.weights
        Autoencoder parameters after the final epoch.

    '''    
    ## Combine the compliance values from all ranks
    if rank==0:
        C_rec=np.empty((size,perrank,1))
    else:
        C_rec=None
    comm.Barrier()
    comm.Gather(C_rank,C_rec,root=0)
    
    ## Determine overall maximum and minimum compliance of training samples and broadcast to all ranks
    if rank==0:
        C_min=np.min(C_rec)
        C_max=np.max(C_rec)
    else:
        C_min=None
        C_max=None
    C_min=comm.bcast(C_min,root=0)
    C_max=comm.bcast(C_max,root=0)
    
    ## Normalize the compliance values so that they belong to [0,1], allowing the surrogate model network to reproduce them 
    C_rank=0.1+(C_rank-C_min)*0.8/(C_max-C_min)
    
    ## Build a model from Encoder and discriminator called Generator, where only the encoder is trainable
    Design_space=Input((X_rank.shape[1],X_rank.shape[2]))
    Latent_space=Input(Discriminator.input.shape[1])
    Disl=Model(Latent_space,Discriminator(Latent_space))
    Disl.trainable=False
    Discriminator.trainable=True
    Generator=Model(Design_space,Disl(Encoder(Design_space)))
    
    ## Build a combined model of Encoder and surrgate model network
    Surr=Model(Design_space,surrogate(Encoder(Design_space)))
    
    
    ## The likelyhood fro training discriminator or generator in each batch
    prob_dis=0.25
    
    ## The number of batches for each epoch and the corresponding batch size in each processor core (rank)
    num_batches=10
    batch_size_perrank=int(perrank/num_batches)
    
    
    ## Initialize parameters for using Adam when updating network parameters
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    eta=0.001
    eta_surr=eta*prob_dis
    m=None
    v=None
    
    m_dis=None
    v_dis=None
    
    m_gen=None
    v_gen=None
    
    m_surr=None
    v_surr=None
    
    gen_start=False
    dis_start=False
    
    ## Initialize the optimizer. As Adam has to be implemented seperately, standard stochastic gradeint descent is used
    if rank==0:
        optimizer=SGD(learning_rate=eta,momentum=0.0)        
        optimizer_surr=SGD(learning_rate=eta_surr,momentum=0.0)
    
    ## Prepare array of indecies for later reshuffleing 
    Index=np.arange(perrank)
    
    ## Wait for all ranks to catch up
    comm.Barrier()
    
    ## Start training the networks
    for epoch in range(n_epochs): 
        
        ##Generate a random order of training samples
        np.random.shuffle(Index)
        
        ## Determine if training with single batch should be performed for achieving convergence in later part of trainig
        if epoch+1>0.9*n_epochs:
            num_batches=1
            batch_size_perrank=perrank
            
        ## Start going through each separate batch
        for batch in range(num_batches):
            
            ## Get training samples and corresponding weights and compliance values for batch
            X_batch=np.copy(X_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            W_batch=np.copy(W_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            C_batch=np.copy(C_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            
            ## Update network parameters for all ranks so they are equal to rank=0 by broadcasting rank=0
            if rank==0:
                AE_weights=AE.get_weights()
                Discriminator_weights=Discriminator.get_weights()
                surrogate_weights=surrogate.get_weights()
            else:
                AE_weights=None
                Discriminator_weights=None
                surrogate_weights=None
            AE_weights=comm.bcast(AE_weights,root=0)
            Discriminator_weights=comm.bcast(Discriminator_weights,root=0)
            surrogate_weights=comm.bcast(surrogate_weights,root=0)
            AE.set_weights(AE_weights)
            Discriminator.set_weights(Discriminator_weights)
            surrogate.set_weights(surrogate_weights)
            
            ## Get the gradient for encoder and decoder respective the reconstruction error (mean squared error)
            ## Get loss function
            with tf.GradientTape() as tape:
                X_batch_pred=AE(X_batch)
                loss_batch=K.mean((X_batch-X_batch_pred)**2*W_batch)/size
            ## Get gradient of loss function
            grad=np.array(tape.gradient(loss_batch,AE.trainable_weights),dtype=object)  
            ## Combine gradients over all ranks
            Gradients=comm.gather(grad,root=0)
            if rank==0:
                Grad=np.sum(Gradients,0)
                
                ## Use Adam to update parameters of autoencoder in rank=0
                if epoch==0 and batch==0:
                    m=(1-betam)*Grad            
                    v=(1-betav)*Grad*Grad
                else:
                    m=betam*m+(1-betam)*Grad             
                    v=betav*v+(1-betav)*Grad*Grad
                mh=m/(1-betamh)
                vh=v/(1-betavh)
                grad_diff=(1/(vh**0.5+1e-8)*mh).tolist()
                optimizer.apply_gradients(zip(grad_diff,AE.trainable_weights))
                
            ## Broadcast rank=0 autoencoder parameters and update remaining ranks
            if rank==0:
                AE_weights=AE.get_weights()
            else:
                AE_weights=None
            AE_weights=comm.bcast(AE_weights,root=0)
            AE.set_weights(AE_weights)
            
            ## Wait for all ranks to catch up
            comm.Barrier()
            
            ## Train the surrogate model and encoder
            ## Get the loss function (mean squared error between rel and predicted compliance values)
            with tf.GradientTape() as tape:
                C_batch_pred=Surr(X_batch)
                loss_batch=K.mean((C_batch-C_batch_pred)**2)/size
            ## Get the gradient of this loss function of encoder and surrogate model network
            grad_surr=np.array(tape.gradient(loss_batch,Surr.trainable_weights),dtype=object)
            ## Combine gradients over all ranks in rank=0
            Gradients_surr=comm.gather(grad_surr,root=0)
            if rank==0:
                Grad_surr=np.sum(Gradients_surr,0)
                
                ## Use Adam to update parameters of networks in rank=0
                if epoch==0 and batch==0:
                    m_surr=(1-betam)*Grad_surr            
                    v_surr=(1-betav)*Grad_surr*Grad_surr
                else:
                    m_surr=betam*m_surr+(1-betam)*Grad_surr             
                    v_surr=betav*v_surr+(1-betav)*Grad_surr*Grad_surr
                mh_surr=m_surr/(1-betamh)
                vh_surr=v_surr/(1-betavh)
                betamh=betamh*betam
                betavh=betavh*betav
                grad_diff_surr=(1/(vh_surr**0.5+1e-8)*mh_surr).tolist()
                optimizer_surr.apply_gradients(zip(grad_diff_surr,Surr.trainable_weights)) 
            
            ## Broadcast rank=0 network parameters and update remaining ranks
            if rank==0:
                Surr_weights=Surr.get_weights()
            else:
                Surr_weights=None
            Surr_weights=comm.bcast(Surr_weights,root=0)
            Surr.set_weights(Surr_weights)
            
            ## Wait for all ranks to catch up
            comm.Barrier()
            
            ## Generate random number in rank 0 and broadcast it to other ranks
            if rank==0:
                prob_rand=np.random.rand()
            else:
                prob_rand=None
            prob_rand=comm.bcast(prob_rand,root=0) 
            
            ## Check if discriminator or generator have to be trained
            if np.abs(0.5-prob_rand)>(0.5-prob_dis):
                if prob_rand<prob_dis:
                    ## Train the generator (that means only encoder parameters)
                    ## Get loss function
                    with tf.GradientTape() as tape:
                        Gen_batch_pred=Generator(X_batch)
                        loss_batch_gen=K.mean(K.log(1+1e-5-Gen_batch_pred))/size
                    ## Get gradient of loss function in respect to network
                    grad_gen=np.array(tape.gradient(loss_batch_gen,Generator.trainable_weights),dtype=object)
                    ## Combine gradeints over all ranks in rank=0
                    Gradients_gen=comm.gather(grad_gen,root=0)
                    if rank==0:
                        Grad_gen=np.sum(Gradients_gen,0)
                        
                        ## Use Adam to update parameters of generator(=encoder) in rank=0 
                        if gen_start==False:
                            m_gen=(1-betam)*Grad_gen            
                            v_gen=(1-betav)*Grad_gen*Grad_gen
                            gen_start=True
                        else:
                            m_gen=betam*m_gen+(1-betam)*Grad_gen             
                            v_gen=betav*v_gen+(1-betav)*Grad_gen*Grad_gen
                        mh_gen=m_gen/(1-betamh)
                        vh_gen=v_gen/(1-betavh)
                        grad_diff_gen=(1/(vh_gen**0.5+1e-8)*mh_gen).tolist()
                        optimizer.apply_gradients(zip(grad_diff_gen,Generator.trainable_weights))
                        
                if 1-prob_rand<prob_dis:
                    ## Train the discriminaor
                    ## Get encoded training samples
                    Z_batch=Encoder.predict(X_batch)
                    ## Get random samples generated according to desired latent space distribution (uniform)
                    Z_rand=np.random.rand(Z_batch.shape[0],Z_batch.shape[1])
                    
                    ## Get loss function
                    with tf.GradientTape() as tape:
                        Dis_batch=Discriminator(Z_batch)
                        Dis_rand=Discriminator(Z_rand)
                        loss_batch_dis=(K.mean(K.log(Dis_batch+1e-5))+K.mean(K.log(1+1e-5-Dis_rand)))/(2*size)
                    ## Get gradient of loss function in respect to disciminator parameters
                    grad_dis=np.array(tape.gradient(loss_batch_dis,Discriminator.trainable_weights),dtype=object)
                    ## Combine gradients over all ranks in rank=0
                    Gradients_dis=comm.gather(grad_dis,root=0)
                    if rank==0:
                        Grad_dis=np.sum(Gradients_dis,0)
                        
                        ## Use Adam to update discriminator parameters
                        if dis_start==False:
                            m_dis=(1-betam)*Grad_dis            
                            v_dis=(1-betav)*Grad_dis*Grad_dis
                            dis_start=True
                        else:
                            m_dis=betam*m_dis+(1-betam)*Grad_dis             
                            v_dis=betav*v_dis+(1-betav)*Grad_dis*Grad_dis
                        mh_dis=m_dis/(1-betamh)
                        vh_dis=v_dis/(1-betavh)
                        grad_diff_dis=(1/(vh_dis**0.5+1e-8)*mh_dis).tolist()
                        optimizer.apply_gradients(zip(grad_diff_dis,Discriminator.trainable_weights)) 
        ## Wait for all ranks to catch up
        comm.Barrier()
    comm.Barrier()
    ## Update autoencoder network parameters once again (only really necessary if last batch included generator training) and brodcast
    if rank==0:
        AE_weights=AE.get_weights()
    else:
        AE_weights=None
    AE_weights=comm.bcast(AE_weights,root=0)   
    
    ## Return autoencoder parameters
    return AE_weights

def train_autoencoder(AE,X_rank,W_rank,comm,rank,size,perrank,n_epochs):  
    '''
    Trains an autoencoder without any additional networks.

    Parameters
    ----------
    AE : Model
        Autoencoder constisting out of encoder and decoder.
    X_rank : perrank*nely*nelx float
        perrank training samples for the autoencoder.
    W_rank : perrank*nely*nelx float
        Weights assigned to each training sample.
    comm : MPI_WORLD
        Envrionment to communicate with different processors.
    rank : int
        The processor used here.
    size : int
        Total number of processors.
    perrank : int
        Number of training samples in each rank.
    n_epochs : int
        Number of training epochs.

    Returns
    -------
    AE_weights : Model.weights
        Autoencoder parameters after the final epoch.

    '''
    num_batches=10
    batch_size_perrank=int(perrank/num_batches)
    
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    eta=0.001
    m=None
    v=None
    Index=np.arange(perrank)
    
    if rank==0:
        optimizer=SGD(learning_rate=eta,momentum=0.0)
    
    comm.Barrier()
    for epoch in range(n_epochs):   
        np.random.shuffle(Index)
        if epoch+1>0.9*n_epochs:
            num_batches=1
            batch_size_perrank=perrank
        for batch in range(num_batches):
            X_batch=np.copy(X_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            W_batch=np.copy(W_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            if rank==0:
                AE_weights=AE.get_weights()
            else:
                AE_weights=None
            AE_weights=comm.bcast(AE_weights,root=0)
            AE.set_weights(AE_weights)
            
            with tf.GradientTape() as tape:
                X_batch_pred=AE(X_batch)
                loss_batch=K.mean((X_batch-X_batch_pred)**2*W_batch)/size
            grad=np.array(tape.gradient(loss_batch,AE.trainable_weights),dtype=object)    
            Gradients=comm.gather(grad,root=0)
            if rank==0:
                Grad=np.sum(Gradients,0)
                if epoch==0 and batch==0:
                    m=(1-betam)*Grad            
                    v=(1-betav)*Grad*Grad
                else:
                    m=betam*m+(1-betam)*Grad             
                    v=betav*v+(1-betav)*Grad*Grad
                mh=m/(1-betamh)
                vh=v/(1-betavh)
                betamh=betamh*betam
                betavh=betavh*betav
                grad_diff=(1/(vh**0.5+1e-8)*mh).tolist()
                optimizer.apply_gradients(zip(grad_diff,AE.trainable_weights))  
        comm.Barrier()
    comm.Barrier()
    if rank==0:
        AE_weights=AE.get_weights()
    else:
        AE_weights=None
    AE_weights=comm.bcast(AE_weights,root=0)
    return AE_weights

def train_AAE(AE,Encoder,Discriminator,X_rank,W_rank,comm,rank,size,perrank,n_epochs):  
    '''
    Trains an adversarial autoencoder which includes a discriminator network.

    Parameters
    ----------
    AE : Model
        Autoencoder constisting out of encoder and decoder.
    Encoder : Model
        Encoder part of the autoencoder AE.
    Discriminator : Model
        The disriminator network.
    X_rank : perrank*nely*nelx float
        perrank training samples for the autoencoder.
    W_rank : perrank*nely*nelx float
        Weights assigned to each training sample.
    comm : MPI_WORLD
        Envrionment to communicate with different processors.
    rank : int
        The processor used here.
    size : int
        Total number of processors.
    perrank : int
        Number of training samples in each rank.
    n_epochs : int
        Number of training epochs.

    Returns
    -------
    AE_weights : Model.weights
        Autoencoder parameters after the final epoch.

    '''
    Design_space=Input((X_rank.shape[1],X_rank.shape[2]))
    Latent_space=Input(Discriminator.input.shape[1])
    Disl=Model(Latent_space,Discriminator(Latent_space))
    Disl.trainable=False
    Discriminator.trainable=True
    Generator=Model(Design_space,Disl(Encoder(Design_space)))
    
    prob_dis=0.25
    
    num_batches=10
    batch_size_perrank=int(perrank/num_batches)
    
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    eta=0.001
    m=None
    v=None
    
    m_dis=None
    v_dis=None
    
    m_gen=None
    v_gen=None
    
    gen_start=False
    dis_start=False
    
    Index=np.arange(perrank)
    if rank==0:
        optimizer=SGD(learning_rate=eta,momentum=0.0)
    
    comm.Barrier()
    for epoch in range(n_epochs): 
        np.random.shuffle(Index)
        if epoch+1>0.9*n_epochs:
            num_batches=1
            batch_size_perrank=perrank
        for batch in range(num_batches):
            X_batch=np.copy(X_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            W_batch=np.copy(W_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            if rank==0:
                AE_weights=AE.get_weights()
                Discriminator_weights=Discriminator.get_weights()
            else:
                AE_weights=None
                Discriminator_weights=None
            AE_weights=comm.bcast(AE_weights,root=0)
            Discriminator_weights=comm.bcast(Discriminator_weights,root=0)
            AE.set_weights(AE_weights)
            Discriminator.set_weights(Discriminator_weights)
            
            with tf.GradientTape() as tape:
                X_batch_pred=AE(X_batch)
                loss_batch=K.mean((X_batch-X_batch_pred)**2*W_batch)/size
            grad=np.array(tape.gradient(loss_batch,AE.trainable_weights),dtype=object)
            Gradients=comm.gather(grad,root=0)
            if rank==0:
                Grad=np.sum(Gradients,0)
                if epoch==0 and batch==0:
                    m=(1-betam)*Grad            
                    v=(1-betav)*Grad*Grad
                else:
                    m=betam*m+(1-betam)*Grad             
                    v=betav*v+(1-betav)*Grad*Grad
                mh=m/(1-betamh)
                vh=v/(1-betavh)
                betamh=betamh*betam
                betavh=betavh*betav
                grad_diff=(1/(vh**0.5+1e-8)*mh).tolist()
                optimizer.apply_gradients(zip(grad_diff,AE.trainable_weights)) 
            if rank==0:
                AE_weights=AE.get_weights()
            else:
                AE_weights=None
            AE_weights=comm.bcast(AE_weights,root=0)
            AE.set_weights(AE_weights)
            comm.Barrier()
            if rank==0:
                prob_rand=np.random.rand()
            else:
                prob_rand=None
            prob_rand=comm.bcast(prob_rand,root=0) 
            if np.abs(0.5-prob_rand)>(0.5-prob_dis):
                if prob_rand<prob_dis:
                    with tf.GradientTape() as tape:
                        Gen_batch_pred=Generator(X_batch)
                        loss_batch_gen=K.mean(K.log(1+1e-5-Gen_batch_pred))/size
                    grad_gen=np.array(tape.gradient(loss_batch_gen,Generator.trainable_weights),dtype=object)
                    Gradients_gen=comm.gather(grad_gen,root=0)
                    if rank==0:
                        Grad_gen=np.sum(Gradients_gen,0)
                        if gen_start==False:
                            m_gen=(1-betam)*Grad_gen            
                            v_gen=(1-betav)*Grad_gen*Grad_gen
                            gen_start=True
                        else:
                            m_gen=betam*m_gen+(1-betam)*Grad_gen             
                            v_gen=betav*v_gen+(1-betav)*Grad_gen*Grad_gen
                        mh_gen=m_gen/(1-betamh)
                        vh_gen=v_gen/(1-betavh)
                        grad_diff_gen=(1/(vh_gen**0.5+1e-8)*mh_gen).tolist()
                        optimizer.apply_gradients(zip(grad_diff_gen,Generator.trainable_weights))
                if 1-prob_rand<prob_dis:
                    Z_batch=Encoder.predict(X_batch)
                    Z_rand=np.random.rand(Z_batch.shape[0],Z_batch.shape[1])
                    with tf.GradientTape() as tape:
                        Dis_batch=Discriminator(Z_batch)
                        Dis_rand=Discriminator(Z_rand)
                        loss_batch_dis=(K.mean(K.log(Dis_batch+1e-5))+K.mean(K.log(1+1e-5-Dis_rand)))/(2*size)
                    grad_dis=np.array(tape.gradient(loss_batch_dis,Discriminator.trainable_weights),dtype=object)
                    Gradients_dis=comm.gather(grad_dis,root=0)
                    if rank==0:
                        Grad_dis=np.sum(Gradients_dis,0)
                        if dis_start==False:
                            m_dis=(1-betam)*Grad_dis            
                            v_dis=(1-betav)*Grad_dis*Grad_dis
                            dis_start=True
                        else:
                            m_dis=betam*m_dis+(1-betam)*Grad_dis             
                            v_dis=betav*v_dis+(1-betav)*Grad_dis*Grad_dis
                        mh_dis=m_dis/(1-betamh)
                        vh_dis=v_dis/(1-betavh)
                        grad_diff_dis=(1/(vh_dis**0.5+1e-8)*mh_dis).tolist()
                        optimizer.apply_gradients(zip(grad_diff_dis,Discriminator.trainable_weights)) 
        comm.Barrier()
    comm.Barrier()
    if rank==0:
        AE_weights=AE.get_weights()
    else:
        AE_weights=None
    AE_weights=comm.bcast(AE_weights,root=0)   
    return AE_weights

def train_autoencoder_surr(AE,Encoder,surrogate,X_rank,W_rank,C_rank,comm,rank,size,perrank,n_epochs): 
    '''
    Trains an autoencoder with an additional surrogate model network.

    Parameters
    ----------
    AE : Model
        Autoencoder constisting out of encoder and decoder.
    Encoder : Model
        Encoder part of the autoencoder AE.
    surrogate : Model
        The surrogate model network.
    X_rank : perrank*nely*nelx float
        perrank training samples for the autoencoder.
    W_rank : perrank*nely*nelx float
        Weights assigned to each training sample.
    C_rank : perrank float
        The compliance of each training sample.
    comm : MPI_WORLD
        Envrionment to communicate with different processors.
    rank : int
        The processor used here.
    size : int
        Total number of processors.
    perrank : int
        Number of training samples in each rank.
    n_epochs : int
        Number of training epochs.

    Returns
    -------
    AE_weights : Model.weights
        Autoencoder parameters after the final epoch.

    '''
    if rank==0:
        C_rec=np.empty((size,perrank,1))
    else:
        C_rec=None
    comm.Barrier()
    comm.Gather(C_rank,C_rec,root=0)
    if rank==0:
        C_min=np.min(C_rec)
        C_max=np.max(C_rec)
    else:
        C_min=None
        C_max=None
    C_min=comm.bcast(C_min,root=0)
    C_max=comm.bcast(C_max,root=0)
    
    C_rank=0.1+(C_rank-C_min)*0.8/(C_max-C_min)
    
    
    Design_space=Input((X_rank.shape[1],X_rank.shape[2]))
    Surr=Model(Design_space,surrogate(Encoder(Design_space)))
    
    num_batches=10
    batch_size_perrank=int(perrank/num_batches)
    
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    eta=0.001
    eta_surr=eta*0.25
    m=None
    v=None
    
    m_surr=None
    v_surr=None
    
    
    Index=np.arange(perrank)
    if rank==0:
        optimizer=SGD(learning_rate=eta,momentum=0.0)        
        optimizer_surr=SGD(learning_rate=eta_surr,momentum=0.0)
    
    comm.Barrier()
    for epoch in range(n_epochs): 
        np.random.shuffle(Index)
        if epoch+1>0.9*n_epochs:
            num_batches=1
            batch_size_perrank=perrank
        for batch in range(num_batches):
            X_batch=np.copy(X_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            W_batch=np.copy(W_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            C_batch=np.copy(C_rank[Index[batch*batch_size_perrank:(batch+1)*batch_size_perrank],:])
            if rank==0:
                AE_weights=AE.get_weights()
                surrogate_weights=surrogate.get_weights()
            else:
                AE_weights=None
                surrogate_weights=None
            AE_weights=comm.bcast(AE_weights,root=0)
            surrogate_weights=comm.bcast(surrogate_weights,root=0)
            AE.set_weights(AE_weights)
            surrogate.set_weights(surrogate_weights)
            
            with tf.GradientTape() as tape:
                X_batch_pred=AE(X_batch)
                loss_batch=K.mean((X_batch-X_batch_pred)**2*W_batch)/size
            grad=np.array(tape.gradient(loss_batch,AE.trainable_weights),dtype=object)            
            Gradients=comm.gather(grad,root=0)
            if rank==0:
                Grad=np.sum(Gradients,0)
                if epoch==0 and batch==0:
                    m=(1-betam)*Grad            
                    v=(1-betav)*Grad*Grad
                else:
                    m=betam*m+(1-betam)*Grad             
                    v=betav*v+(1-betav)*Grad*Grad
                mh=m/(1-betamh)
                vh=v/(1-betavh)
                grad_diff=(1/(vh**0.5+1e-8)*mh).tolist()
                optimizer.apply_gradients(zip(grad_diff,AE.trainable_weights)) 
            if rank==0:
                AE_weights=AE.get_weights()
            else:
                AE_weights=None
            AE_weights=comm.bcast(AE_weights,root=0)
            AE.set_weights(AE_weights)
            comm.Barrier()
            #Surrogate
            with tf.GradientTape() as tape:
                C_batch_pred=Surr(X_batch)
                loss_batch=K.mean((C_batch-C_batch_pred)**2)/size
            grad_surr=np.array(tape.gradient(loss_batch,Surr.trainable_weights),dtype=object)            
            Gradients_surr=comm.gather(grad_surr,root=0)
            if rank==0:
                Grad_surr=np.sum(Gradients_surr,0)
                if epoch==0 and batch==0:
                    m_surr=(1-betam)*Grad_surr            
                    v_surr=(1-betav)*Grad_surr*Grad_surr
                else:
                    m_surr=betam*m_surr+(1-betam)*Grad_surr             
                    v_surr=betav*v_surr+(1-betav)*Grad_surr*Grad_surr
                mh_surr=m_surr/(1-betamh)
                vh_surr=v_surr/(1-betavh)
                betamh=betamh*betam
                betavh=betavh*betav
                grad_diff_surr=(1/(vh_surr**0.5+1e-8)*mh_surr).tolist()
                optimizer_surr.apply_gradients(zip(grad_diff_surr,Surr.trainable_weights)) 
            if rank==0:
                Surr_weights=Surr.get_weights()
            else:
                Surr_weights=None
            Surr_weights=comm.bcast(Surr_weights,root=0)
            Surr.set_weights(Surr_weights)
        comm.Barrier()
    comm.Barrier()
    if rank==0:
        AE_weights=AE.get_weights()
    else:
        AE_weights=None
    AE_weights=comm.bcast(AE_weights,root=0)   
    return AE_weights