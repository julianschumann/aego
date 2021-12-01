import numpy as np
from mpi4py import MPI
import os
from SIMP import TO_SIMP, make_Conn_matrix
import time
from keras.models import load_model
from bayesian_optimization import kriging, ExImp, corr_matrix


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
        ## Enforce a design with sparse densities
        X_out[i,:,:], C[i] = TO_SIMP(X_out[i,:,:], volfrac, penal, beta, epsilon_2, max_move, E0, nu, Iar, cMat, True , void, avoid, 0, 10)
    return X_out,C


## Implent multiprocessing, with size processors, where rank is the one currently executing this file
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()




## Load the decoded latent space over which optimization is to be performed
Decoder_model='Sample_data/edim=100_pre=1_AAE=1_surr=1_sdf_case=3_Decoder.h5'
Decoder=load_model(Decoder_model)

## Set type of encoding
typ=Decoder_model[-21:-18] ## Only works for case<10
## Set number of optimization steps in latent space
opt_it=25
## Get problem dimensionality
[nely,nelx]=Decoder.output_shape[1:]
## Set parameters needed for cost function (Compliance minimization)
volfrac=0.4
Iar, cMat= make_Conn_matrix(nelx,nely)
void=get_void(nely, nelx)

## Get the dimensionality of the latent space over which optimization is to be performed
edim=Decoder.input_shape[1]

## Set the parameters for differential evolution (needed for ptimization of aqusition function)
multiplyer=0.6
prob_change=0.9



## Perform the Bayesian optimization

## Set the number of initial samples
train_samples=10*edim
train_samples_perrank=int(train_samples/size)
train_samples=train_samples_perrank*size

## Initialize Gaussian process model parameters
p=np.zeros(edim)
theta=np.ones(edim)

## Set number of itertions
bayes_iteration=edim*3
comm.Barrier()

start_time=time.time()

## Generate the number of training samples and evaluate them
Z0_rank=np.random.rand(train_samples_perrank,edim)
X0_rank,C0_rank=evaluate_design(Z0_rank,Decoder,volfrac,Iar,cMat,void,opt_it,typ)

## Share the samples across processor cores/ranks
if rank==0:
    Z0_rec=np.empty((size,train_samples_perrank,edim))
    X0_rec=np.empty((size,train_samples_perrank,nely,nelx))
    C0_rec=np.empty((size,train_samples_perrank))
else:
    Z0_rec=None
    X0_rec=None
    C0_rec=None
comm.Barrier()
comm.Gather(Z0_rank,Z0_rec,root=0)
comm.Gather(X0_rank,X0_rec,root=0)
comm.Gather(C0_rank,C0_rec,root=0)
if rank==0:
    Z0=Z0_rec.reshape((train_samples,edim))
    X0=X0_rec.reshape((train_samples,nely,nelx))
    C0=C0_rec.reshape((train_samples,1))
else:
    Z0=None
    X0=None
    C0=None
Z0=comm.bcast(Z0,root=0)
X0=comm.bcast(X0,root=0)
C0=comm.bcast(C0,root=0)

## Start the iterative optimization process
for ib in range(bayes_iteration):
    if rank==0:
        print('        Iteration {}'.format(ib))
    start_time_it=time.time()
    start_time_its=time.time()
    ## That the weight to focus on ...
    if ib<bayes_iteration-100:
        ## ...exploration
        weight_explore=10
    else:
        ## ...exploitation
        weight_explore=0  
        
    ## Set number of optimization steps when generation Gaussian process model parameters
    if ib==0:
        k_iter=50
        ## Get Gaussian process model parameters
        theta,p=kriging(Z0, C0,rank,size,[],k_iter)
    else:
        if np.mod(ib,100)==0:
            k_iter=50
        else:
            k_iter=10
        para_old=np.concatenate((theta,p-0.5),0)
        ## Get Gaussian process model parameters
        theta,p=kriging(Z0, C0,rank,size,para_old,k_iter)
    ## Get the Gaussian process model corealation matrix for optimized parameters
    K=corr_matrix(Z0, theta[:,np.newaxis], p[:,np.newaxis])[:,:,0]    
    ## Get the inverse of the correlation matrix (adapt it when the matrix is singular)            
    inverse_failed=True
    while inverse_failed:
        try: 
            Kinv=np.linalg.inv(K) 
            inverse_failed=False
        except np.linalg.LinAlgError:
            K=K-np.identity(len(C0))*1e-4
    stop_time_its=time.time()
    if rank==0:
        time_needed=stop_time_its-start_time_its
        print('            Time needed for model training: {:10.1f}s'.format(time_needed))
    start_time_its=time.time()    
    ## Optimize aqusition function using differential evolution
    EI_num_pop_perrank=int(np.ceil(2.5*edim/size))
    EI_num_pop=size*EI_num_pop_perrank
    
    ## Initial generation is generated and evaluated
    Zei_rank=np.random.rand(EI_num_pop_perrank,edim)
    EI_rank=ExImp(Zei_rank, theta, p, Z0, C0, Kinv, weight_explore)
    ## Initial generation is shared over all ranks
    if rank==0:
        Zei_rec=np.empty((size,EI_num_pop_perrank,edim))
        EI_rec=np.empty((size,EI_num_pop_perrank))
    else:
        Zei_rec=None
        EI_rec=None
    comm.Barrier()
    comm.Gather(Zei_rank,Zei_rec,root=0)
    comm.Gather(EI_rank,EI_rec,root=0)
    if rank==0:
        Zei=Zei_rec.reshape((EI_num_pop,edim))
        EI=EI_rec.reshape(EI_num_pop)
    else:
        Zei=None
        EI=None 
    Zei=comm.bcast(Zei,root=0)
    EI=comm.bcast(EI,root=0)
    loop_ei=0
    loop_ei_max=500
    ## Generations are evolved
    while loop_ei<loop_ei_max:
        Zei_rank=Zei[rank*EI_num_pop_perrank:(rank+1)*EI_num_pop_perrank,:]
        EI_rank=EI[rank*EI_num_pop_perrank:(rank+1)*EI_num_pop_perrank]
    
        ## Reproduction between differnt individuals from the population is perforemd
        test_case=np.floor(np.random.rand(EI_num_pop_perrank,3)*(EI_num_pop-1e-7)).astype('int')
        Za_rank=np.copy(Zei[test_case[:,0],:])
        Zb_rank=np.copy(Zei[test_case[:,1],:])
        Zc_rank=np.copy(Zei[test_case[:,2],:])
        Zcom_rank=Za_rank+multiplyer*(Zb_rank-Zc_rank)
        
        ## Crossover between child and parent is performed
        prob=np.random.rand(EI_num_pop_perrank,edim)
        Zcom_rank[prob>prob_change]=np.copy(Zei_rank[prob>prob_change])
        
        ## Boundaries of design are enforced
        Zcom_rank[Zcom_rank<0]=0
        Zcom_rank[Zcom_rank>1]=1 
        
        ## Selection between child (has to be evaluated first) and parent is performed
        EI_compare=ExImp(Zcom_rank, theta, p, Z0, C0, Kinv, weight_explore)
        EI_rank=np.minimum(EI_rank,EI_compare)
        Zei_rank[EI_compare<=EI_rank,:]=Zcom_rank[EI_compare<=EI_rank,:]
        
        ## New population is shared between all ranks
        if rank==0:
            Zei_rec=np.empty((size,EI_num_pop_perrank,edim))
            EI_rec=np.empty((size,EI_num_pop_perrank))
        else:
            Zei_rec=None
            EI_rec=None
        comm.Barrier()
        comm.Gather(Zei_rank,Zei_rec,root=0)
        comm.Gather(EI_rank,EI_rec,root=0)
        if rank==0:
            Zei=Zei_rec.reshape((EI_num_pop,edim))
            EI=EI_rec.reshape(EI_num_pop)
        else:
            Zei=None
            EI=None 
        Zei=comm.bcast(Zei,root=0)
        EI=comm.bcast(EI,root=0)
        loop_ei=loop_ei+1
    stop_time_its=time.time()
    if rank==0:
        time_needed=stop_time_its-start_time_its
        print('            Time needed for optimizing acqusition function: {:10.1f}s'.format(time_needed))
    start_time_its=time.time()
    
    ## The training samples are updated with the one having the highest expected improvement
    if rank==0:
        jmin=np.argmin(EI)
        Z_new=Zei[[jmin],:]
        X_new,C_new=evaluate_design(Z_new,Decoder,volfrac,Iar,cMat,void,opt_it,typ)
        C0=np.concatenate((C0,C_new[:,np.newaxis]),0)
        Z0=np.concatenate((Z0,Z_new),0)
        X0=np.concatenate((X0,X_new))
    # The new samples are shared across ranks
    Z0=comm.bcast(Z0,root=0)
    X0=comm.bcast(X0,root=0)
    C0=comm.bcast(C0,root=0)    
    stop_time_it=time.time()
    stop_time_its=time.time()
    if rank==0:
        time_needed=stop_time_its-start_time_its
        print('            Time needed for updating data: {:10.1f}s'.format(time_needed))
    if rank==0:
        time_needed=stop_time_it-start_time_it
        print('        Time needed for iteration: {:10.1f}s'.format(time_needed))
comm.Barrier()

## Best training sample is determined
if rank==0:
    Z_min=Z0[[np.argmin(C0)],:]
    ## Postprocessing is performed on best training sample
    X_final,F_final=evaluate_design(Z_min,Decoder,volfrac,Iar,cMat,void,300,typ)
    X_final=X_final[0,:,:]
    F_final=F_final[0]
else:
    Z_min=None 
    X_final=None 
    F_final=None 
    data=None

## Post process result of the optimization is saved is saved
comm.Barrier()
stop_time=time.time()
if rank==0:
    data=[X_final,F_final,stop_time-start_time]
    np.save('Sample_data/BO_test.npy',np.array(data))

                    


