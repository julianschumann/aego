import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model 
from keras.layers.advanced_activations import LeakyReLU
from mpi4py import MPI
from keras.optimizers import SGD
import keras.backend as K 
import time


def mapping(dim,edim):
    n=5000
    Z0=np.random.rand(n,edim)*2-1
    Z0[0,:]=0.5
    W1=(np.random.rand(edim,dim)-0.5)*0.4
    X1=np.tanh(np.dot(Z0,W1))
    N=np.log(0.5*X1[[0],:]+0.5)/np.log(0.625)
    X2=2*((X1+1)/2)**(1/N)-1
    return X2

def cost_opt(X,opt_it,X0):
    M=np.zeros(X.shape)
    V=np.zeros(X.shape)
    eta=0.001
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    for i in range(opt_it):
        G=gradient(X,X0)
        M=betam*M+(1-betam)*G
        V=betav*V+(1-betav)*G**2
        Mh=M/(1-betamh)
        Vh=V/(1-betavh)
        betamh=betamh*betam
        betavh=betavh*betav
        D=eta*Mh/(Vh**0.5 +1e-8)
        X=X-D
        X=np.clip(X,-1,1)
    return cost(X,X0)

def cost_dec(Z,decoder,opt_it,X0):
    X=decoder.predict(Z)
    M=np.zeros(X.shape)
    V=np.zeros(X.shape)
    eta=0.001
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    for i in range(opt_it):
        G=gradient(X,X0)
        M=betam*M+(1-betam)*G
        V=betav*V+(1-betav)*G**2
        Mh=M/(1-betamh)
        Vh=V/(1-betavh)
        betamh=betamh*betam
        betavh=betavh*betav
        D=eta*Mh/(Vh**0.5 +1e-8)
        X=X-D
        X=np.clip(X,-1,1)
    return cost(X,X0)

def cost(X,X0):
    C0=np.zeros(len(X))
    for i in range(len(X)):
        C0[i]=np.min(np.sum((X0[1:,:]-X[[i],:])**2,1))
    Xn=np.sum((X-0.25)**2,1)
    Cn=(1-np.exp(-5*Xn))*(0.2+np.exp(-5*Xn))*5
    return (C0+1)*Cn

def gradient(X,X0):
    Xn=np.sum((X-0.25)**2,1)
    Cn=(1-np.exp(-5*Xn))*(0.2+np.exp(-5*Xn))*5
    dXndX=2*(X-0.25)
    dCndXn=50*np.exp(-10*Xn)-20*np.exp(-5*Xn)
    Gn=dCndXn[:,np.newaxis]*dXndX
    G0=np.zeros(X.shape)
    C0=np.zeros(len(X))
    for i in range(len(X)):
        jmin=np.argmin(np.sum((X0[1:,:]-X[[i],:])**2,1))
        G0[i,:]=2*(X[i,:]-X0[jmin+1,:])
        C0[i]=np.sum((X0[jmin+1,:]-X[i,:])**2)
    G=Cn[:,np.newaxis]*G0+(C0[:,np.newaxis]+1)*Gn
    return G

def train_autoencoder(AE,X_rank,rank,size,perrank,n_epochs):   
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
            if rank==0:
                AE_weights=AE.get_weights()
            else:
                AE_weights=None
            AE_weights=comm.bcast(AE_weights,root=0)
            AE.set_weights(AE_weights)
            
            with tf.GradientTape() as tape:
                X_batch_pred=AE(X_batch)
                loss_batch=K.mean((X_batch-X_batch_pred)**2)/size
            grad=np.array(tape.gradient(loss_batch,AE.trainable_weights),dtype=object)              
            Gradient=[None]*len(grad)
            for i in range(len(grad)):
                Gradient[i]=comm.gather(grad[i],root=0)
            # Gradients=comm.gather(grad,root=0)
            if rank==0:
                # Grad=np.sum(Gradients,0)
                Grad=np.sum(Gradient,1)
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(rank)

dim=500
Edim=[10,5]

# ensure other local minima not to close to global optimum
if rank==0:
    X0=mapping(dim,10)    
    Dmin=np.sqrt(np.min(np.sum((X0[1:,:]-0.25)**2,1)))
    while Dmin<1:        
        X0=mapping(dim,10)    
        Dmin=np.sqrt(np.min(np.sum((X0[1:,:]-0.25)**2,1)))
    np.save('Results/Local_minima.npy',X0)
else:
    X0=None

comm.Barrier()
X0=comm.bcast(X0,root=0)
perrank=int((len(X0)-1)/size)
X_rank=X0[1+rank*perrank:1+(rank+1)*perrank,:]
opt_it=10
######################################################################
##                   Proposed Method                                ##
###################################################################### 

for edim in Edim:    
    start_time=time.time()
    ## Now train network
    design_space=Input((dim))
    enc=Dense(dim,activation='tanh')(design_space)
    enc=LeakyReLU(alpha=0.3)(enc)
    enc=Dense(dim,activation='tanh')(enc)
    enc=LeakyReLU(alpha=0.3)(enc)
    enc=Dense(dim,activation='tanh')(enc)
    enc=LeakyReLU(alpha=0.3)(enc)
    enc=Dense(edim,activation='sigmoid')(enc)
    encoder=Model(design_space,enc)
    
    latent_space=Input((edim))
    dec=Dense(dim,activation='tanh')(latent_space)
    dec=LeakyReLU(alpha=0.3)(dec)
    dec=Dense(dim,activation='tanh')(dec)
    dec=LeakyReLU(alpha=0.3)(dec)
    dec=Dense(dim,activation='tanh')(dec)
    dec=LeakyReLU(alpha=0.3)(dec)
    dec=Dense(dim,activation='tanh')(dec)
    decoder=Model(latent_space,dec)
    
    AE=Model(design_space,decoder(encoder(design_space)))
          
    AE_weights=train_autoencoder(AE,X_rank,rank,size,perrank,250)
    AE.set_weights(AE_weights)
    
    stop_time=time.time()
    if rank==0:
        print('Step 2: {:1.1f}s'.format(stop_time-start_time))
    start_time=time.time()
    # Optimize in latent space
    comm.Barrier()
    num_pop_perrank=int(np.ceil(10*edim/size))
    num_pop=num_pop_perrank*size
    prob_change=0.9
    multiplyer=0.6
    
    Z_rank=np.random.rand(num_pop_perrank,edim)
    F_rank=cost_dec(Z_rank, decoder, opt_it, X0)
    comm.Barrier()
    if rank==0:
        Z_rec=np.empty((size,num_pop_perrank,edim))
        F_rec=np.empty((size,num_pop_perrank))
    else:
        Z_rec=None
        F_rec=None
    comm.Gather(Z_rank,Z_rec,root=0)
    comm.Gather(F_rank,F_rec,root=0)
    if rank==0:
        Z=Z_rec.reshape((num_pop_perrank*size,edim))
        F=F_rec.reshape(num_pop_perrank*size)
    else:
        Z=None
        F=None                                
    Z=comm.bcast(Z,root=0)
    F=comm.bcast(F,root=0)
    C_ae=np.zeros((num_pop,501))
    C_ae[:,0]=F[:num_pop]
    loop=0
    while loop<500:
        Z_rank=Z[rank*num_pop_perrank:(rank+1)*num_pop_perrank,:]
        F_rank=F[rank*num_pop_perrank:(rank+1)*num_pop_perrank]
    
        test_case=np.floor(np.random.rand(num_pop_perrank,3)*(num_pop-1e-7)).astype('int')
        Za_rank=np.copy(Z[test_case[:,0],:])
        Zb_rank=np.copy(Z[test_case[:,1],:])
        Zc_rank=np.copy(Z[test_case[:,2],:])
        Zcom_rank=Za_rank+multiplyer*(Zb_rank-Zc_rank)
        prob=np.random.rand(num_pop_perrank,edim)
        Zcom_rank[prob>prob_change]=np.copy(Z_rank[prob>prob_change])
        Zcom_rank[Zcom_rank<0]=0
        Zcom_rank[Zcom_rank>1]=1 
        F_compare=cost_dec(Zcom_rank, decoder, opt_it, X0)
        F_rank=np.minimum(F_rank,F_compare)
        Z_rank[F_compare<=F_rank,:]=Zcom_rank[F_compare<=F_rank,:]
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
            Z=Z_rec.reshape((num_pop_perrank*size,edim))
            F=F_rec.reshape(num_pop_perrank*size)
        Z=comm.bcast(Z,root=0)
        F=comm.bcast(F,root=0)
        loop=loop+1    
        C_ae[:,loop]=F[:num_pop]
    
    
    C_min_ae=np.min(C_ae,0)
    if rank==0:
        eval_ae=np.linspace(0,num_pop*500*(opt_it+1),501)
    else:
        eval_ae=None
    
    stop_time=time.time()
    if rank==0:
        print('Step 3: {:1.1f}s'.format(stop_time-start_time))
    start_time=time.time()
    #post opt
    i_post=1000
    X_post_rank=decoder.predict(Z_rank)
    C_rank=np.zeros((num_pop_perrank,i_post+1))
    C_rank[:,0]=cost(X_post_rank,X0)
    M_rank=np.zeros((num_pop_perrank,dim))
    V_rank=np.zeros((num_pop_perrank,dim))
    eta=0.001
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    
    for iteration in range(i_post):
        G_rank=gradient(X_post_rank,X0)
        M_rank=betam*M_rank+(1-betam)*G_rank
        V_rank=betav*V_rank+(1-betav)*G_rank**2
        Mh_rank=M_rank/(1-betamh)
        Vh_rank=V_rank/(1-betavh)
        betamh=betamh*betam
        betavh=betavh*betav
        D_rank=eta*Mh_rank/(Vh_rank**0.5 +1e-8)
        X_post_rank=X_post_rank-D_rank
        X_post_rank=np.clip(X_post_rank,-1,1)
        C_rank[:,iteration+1]=cost(X_post_rank,X0)
        
    
    if rank==0:
        C_post=np.zeros((size,num_pop_perrank,i_post+1))
    else:
        C_post=None
        eval_post=None
        data=None
    comm.Barrier()
    comm.Gather(C_rank,C_post,root=0)
    
    stop_time=time.time()
    if rank==0:
        print('Step 4: {:1.1f}s'.format(stop_time-start_time))
        C_post=np.min(C_post[:,:,opt_it:],(0,1))
        eval_post=np.linspace(0,(i_post-opt_it)*num_pop,i_post+1-opt_it)+eval_ae[-1]
        data=[1000,C_min_ae,C_post,0,eval_ae,eval_post,0]
        np.save('Results/Benchmark_data_edim={}.npy'.format(edim),np.array(data))
######################################################################
##                   Comparison Method                              ##
######################################################################

comm.Barrier()
start_time=time.time()
num_pop_perrank=int(np.ceil(0.5*dim/size))
num_pop=size*num_pop_perrank
prob_change=0.9
multiplyer=0.6

X_rank=np.random.rand(num_pop_perrank,dim)
C_rank=cost_opt(X_rank,opt_it, X0)
comm.Barrier()
if rank==0:
    X_rec=np.empty((size,num_pop_perrank,dim))
    C_rec=np.empty((size,num_pop_perrank))
else:
    X_rec=None
    C_rec=None
comm.Gather(X_rank,X_rec,root=0)
comm.Gather(C_rank,C_rec,root=0)
if rank==0:
    X=X_rec.reshape((num_pop,dim))
    C=C_rec.reshape(num_pop)
else:
    X=None
    C=None                                
X=comm.bcast(X,root=0)
C=comm.bcast(C,root=0)
C_de=np.zeros((num_pop,1001))
C_de[:,0]=C
loop=0
while loop<1000:
    if rank==0:
        print('   Generation {}'.format(loop))
    X_rank=X[rank*num_pop_perrank:(rank+1)*num_pop_perrank,:]
    C_rank=C[rank*num_pop_perrank:(rank+1)*num_pop_perrank]

    test_case=np.floor(np.random.rand(num_pop_perrank,3)*(num_pop-1e-7)).astype('int')
    Xa_rank=np.copy(X[test_case[:,0],:])
    Xb_rank=np.copy(X[test_case[:,1],:])
    Xc_rank=np.copy(X[test_case[:,2],:])
    Xcom_rank=Xa_rank+multiplyer*(Xb_rank-Xc_rank)
    prob=np.random.rand(num_pop_perrank,dim)
    Xcom_rank[prob>prob_change]=np.copy(X_rank[prob>prob_change])
    Xcom_rank[Xcom_rank<0]=0
    Xcom_rank[Xcom_rank>1]=1 
    C_compare=cost_opt(Xcom_rank,opt_it, X0)
    C_rank=np.minimum(C_rank,C_compare)
    X_rank[C_compare<=C_rank,:]=Xcom_rank[C_compare<=C_rank,:]
    if rank==0:
        X_rec=np.empty((size,num_pop_perrank,dim))
        C_rec=np.empty((size,num_pop_perrank))
    else:
        X_rec=None
        C_rec=None
    comm.Gather(X_rank,X_rec,root=0)
    comm.Gather(C_rank,C_rec,root=0)
    if rank==0:
        X=X_rec.reshape((num_pop,dim))
        C=C_rec.reshape(num_pop)
    else:
        X=None
        C=None                                
    X=comm.bcast(X,root=0)
    C=comm.bcast(C,root=0)
    loop=loop+1    
    C_de[:,loop]=C


C_min_de=np.min(C_de,0)
if rank==0:
    eval_de=np.linspace(0,num_pop*1000*(opt_it+1),1001)
else:
    eval_de=None

stop_time=time.time()
if rank==0:
    print('DE: {:1.1f}s'.format(stop_time-start_time))
start_time=time.time()
#post opt
i_post=200
X_post_rank=np.copy(X_rank)
C_rank=np.zeros((num_pop_perrank,i_post+1))
C_rank[:,0]=cost(X_post_rank,X0)
M_rank=np.zeros((num_pop_perrank,dim))
V_rank=np.zeros((num_pop_perrank,dim))
eta=0.001
betam=0.9
betav=0.999
betamh=0.9
betavh=0.999

for it_post in range(i_post):
    G_rank=gradient(X_post_rank,X0)
    M_rank=betam*M_rank+(1-betam)*G_rank
    V_rank=betav*V_rank+(1-betav)*G_rank**2
    Mh_rank=M_rank/(1-betamh)
    Vh_rank=V_rank/(1-betavh)
    betamh=betamh*betam
    betavh=betavh*betav
    D_rank=eta*Mh_rank/(Vh_rank**0.5 +1e-8)
    X_post_rank=X_post_rank-D_rank
    X_post_rank=np.clip(X_post_rank,-1,1)
    C_rank[:,it_post+1]=cost(X_post_rank,X0)
    

if rank==0:
    C_post=np.zeros((size,num_pop_perrank,i_post+1))
else:
    C_post=None
    eval_post=None
    data=None
comm.Barrier()
comm.Gather(C_rank,C_post,root=0)

stop_time=time.time()
if rank==0:
    print('LO: {:1.1f}s'.format(stop_time-start_time))
    C_post=np.min(C_post[:,:,opt_it:],(0,1))
    eval_post=np.linspace(0,(i_post-opt_it)*num_pop,i_post+1-opt_it)+eval_de[-1]
    data=[C_min_de,C_post,eval_de,eval_post,0]
    np.save('Results/Benchmark_test_comp_data.npy',np.array(data))

    

