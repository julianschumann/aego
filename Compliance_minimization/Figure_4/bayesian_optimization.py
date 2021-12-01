import numpy as np
from mpi4py import MPI
from scipy.stats import norm


def corr_matrix(X,theta,p):
    '''
    Builds the correlation matrix R of the gaussian process model for certain parameters

    Parameters
    ----------
    X : n*dim float
        The n samples the Gaussian process model is build upon.
    theta : dim float
        Mutliplier in the Gaussian process model.
    p : dim float
        Exponents in the Gaussian process model.

    Returns
    -------
    R : n*n float
        Correlation matrix in the Gaussian process model.

    '''
    theta=theta[np.newaxis,np.newaxis,:]
    p=p[np.newaxis,np.newaxis,:]
    R=np.exp(-np.sum(theta*np.abs(X[:,np.newaxis,:,np.newaxis]-X[np.newaxis,:,:,np.newaxis])**p,2))
    return R  

def likely(X,C,para): 
    para=np.transpose(para)
    if len(C.shape)==1:
        C=C[:,np.newaxis]
    num_samples=X.shape[0]
    dim=X.shape[1]
    theta=para[:dim,:]
    p=para[dim:,:]+0.5
    K=corr_matrix(X,theta,p)
    Kinv=np.zeros(K.shape)
    for i in range(K.shape[2]):
        inverse_failed=True
        while inverse_failed:
            try: 
                Kinv[:,:,i]=np.linalg.inv(K[:,:,i]) 
                inverse_failed=False
            except np.linalg.LinAlgError:
                K[:,:,i]=K[:,:,i]-np.identity(num_samples)*1e-4
    one=np.ones((num_samples,1))
    mu=np.dot(C.transpose(),np.dot(np.transpose(one),Kinv)[0])/np.dot(one.transpose(),np.dot(np.transpose(one),Kinv)[0])
    sigmasq=np.zeros((1,para.shape[1]))
    for i in range(para.shape[1]):
        hi=C-one*mu[0,i]
        sigmasq[0,i]=np.dot(hi.transpose(),np.dot(Kinv[:,:,i],hi))
    L=num_samples*np.log(np.abs(sigmasq[0,:])+1e-8)+np.log(np.abs(np.linalg.det(K.transpose(2,0,1)))+1e-8)
    return L        

def kriging(X,C,comm,rank,size,para_old,k_iter):
    '''
    Determines the parameters of a Gaussian process model.
    This is done using differential evolution on multiple cores to optimize the likelyhood of the model belonging to the parameters.

    Parameters
    ----------
    X : n*dim float
        The n samples the Gaussian process model is build upon.
    C : n*1 float
        The cost function at the n samples.
    comm : MPI_WORLD
        Envrionment to communicate with different processors.
    rank : int
        The processor used here.
    size : int
        Total number of processors.
    para_old : 2*dim float or []
        If existig, the optimal parameters from the previous iteration.
    k_iter : int
        Number of generations when using differential evolution.

    Returns
    -------
    theta : dim float
        Mutliplier in the Gaussian process model.
    p : dim float
        Exponents in the Gaussian process model.

    '''
    dim=X.shape[1]
    prob_change=0.9
    multiplyer=0.6
    imax=k_iter
    train_samples=int(dim)
    train_samples_perrank=int(np.ceil(train_samples/size))
    train_samples=train_samples_perrank*size
    Para_rank=np.random.rand(train_samples_perrank,2*dim)*4.9+0.1
    if len(para_old)>0 and rank==0:
        Para_rank[0,:]=para_old
    L_rank=likely(X,C,Para_rank)
    if rank==0:
        Para_rec=np.empty((size,train_samples_perrank,2*dim))
        L_rec=np.empty((size,train_samples_perrank))
    else:
        Para_rec=None
        L_rec=None
    comm.Barrier()
    comm.Gather(Para_rank,Para_rec,root=0)
    comm.Gather(L_rank,L_rec,root=0)
    if rank==0:
        Para=Para_rec.reshape((train_samples,2*dim))
        L=L_rec.reshape(train_samples)
    else:
        Para=None
        L=None 
    Para=comm.bcast(Para,root=0)
    L=comm.bcast(L,root=0)
    loop=0
    while loop<imax:
        Para_rank=Para[rank*train_samples_perrank:(rank+1)*train_samples_perrank,:]
        L_rank=L[rank*train_samples_perrank:(rank+1)*train_samples_perrank]
    
        test_case=np.floor(np.random.rand(train_samples_perrank,3)*(train_samples-1e-7)).astype('int')
        Paraa_rank=np.copy(Para[test_case[:,0],:])
        Parab_rank=np.copy(Para[test_case[:,1],:])
        Parac_rank=np.copy(Para[test_case[:,2],:])
        Paracom_rank=Paraa_rank+multiplyer*(Parab_rank-Parac_rank)
        prob=np.random.rand(train_samples_perrank,2*dim)
        Paracom_rank[prob>prob_change]=np.copy(Para_rank[prob>prob_change])
        Paracom_rank[Paracom_rank<0.1]=0.1
        Paracom_rank[Paracom_rank>5]=5
        L_compare=likely(X,C,Paracom_rank)
        L_rank=np.minimum(L_rank,L_compare)
        Para_rank[L_compare<=L_rank,:]=Paracom_rank[L_compare<=L_rank,:]
        if rank==0:
            Para_rec=np.empty((size,train_samples_perrank,2*dim))
            L_rec=np.empty((size,train_samples_perrank))
        else:
            Para_rec=None
            L_rec=None
        comm.Barrier()
        comm.Gather(Para_rank,Para_rec,root=0)
        comm.Gather(L_rank,L_rec,root=0)
        if rank==0:
            Para=Para_rec.reshape((train_samples,2*dim))
            L=L_rec.reshape(train_samples)
        else:
            Para=None
            L=None 
        Para=comm.bcast(Para,root=0)
        L=comm.bcast(L,root=0)
        loop=loop+1
    if rank==0:
        jmin=np.argmin(L)
        theta=Para[jmin,:dim]
        p=Para[jmin,dim:]
    else:
        theta=None
        p=None
    comm.Barrier()
    theta=comm.bcast(theta,root=0)
    p=comm.bcast(p,root=0)
    return theta,p+0.5

def ExImp(Xpos,theta,p,X,C,Rinv,weight_explore):
    '''
    Build an aqusition function, the expected improvement function, for bayesian optimization based on a Gaussian process model [1].
    
    [1]: Donald R Jones, Matthias Schonlau, and William J Welch. Efficient global optimization of expensive black-box functions. Journal of Global optimization, 13(4):455–492, 1998.

    Parameters
    ----------
    Xpos : m*dim
        The m samples at which the expected improvement function is to be evaluated.
    theta : dim float
        Mutliplier in the Gaussian process model.
    p : dim float
        Exponents in the Gaussian process model.
    X : n*dim float
        The n samples the Gaussian process model is build upon.
    C : n*1 float
        The cost function at the n samples.
    Rinv : n*n float
        The inverse of the correlation matrix in the Gaussian process model.
    weight_explore : float
        A factor used to allow to shift focus between exploration (heigh value) and exploitation (low value).

    Returns
    -------
    m float
        The negative value of the expected improvement function (EI has to maximized, and therefore, -EI is minimized).

    '''
    num_samples=len(X)
    one=np.ones((num_samples,1))
    mu=np.dot(np.dot(np.transpose(one),Rinv),C)/(np.dot(np.dot(np.transpose(one),Rinv),one)+1e-8)
    sigmasq=np.dot(np.dot(np.transpose(C-mu*one),Rinv),(C-mu*one))/num_samples
    theta=theta[np.newaxis,np.newaxis,:]
    p=p[np.newaxis,np.newaxis,:]
    k=np.exp(-np.sum(theta*np.abs(Xpos[np.newaxis,:,:]-X[:,np.newaxis,:])**p,2))
    pred=mu+np.dot(np.dot(np.transpose(k),Rinv),(C-one*mu))
    #check how the matrix dimensions work out here
    conf=np.sqrt(np.abs(sigmasq*(1-np.sum(k*np.dot(Rinv,k),0)+(1-np.dot(np.transpose(one),np.dot(Rinv,k)))**2/(np.dot(np.dot(np.transpose(one),Rinv),one)+1e-8))))
    conf=np.transpose(conf)
    Cmin=np.min(C)
    EI=(Cmin-pred-weight_explore)*norm.cdf((Cmin-pred-weight_explore)/(conf+1e-8))+ conf*norm.pdf((Cmin-pred-weight_explore)/(conf+1e-8))
    # give out negative EI, as EI is minimized
    return -EI[:,0]