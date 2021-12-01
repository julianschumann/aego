import numpy as np
from signed_distance_field import simp2sdf
from mpi4py import MPI
import os
from SIMP import TO_SIMP, make_Conn_matrix

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

def get_design(x0,volfrac,avoid_1,avoid_2,void,it_steps,Iar,cMat):
    '''
    The generation of a the training samples, see Appendix I.1 of my master thesis.

    Parameters
    ----------
    x0 : nely*nelx float
        Initial material distribution.
    volfrac : float
        Part of volume that has to be filled..
    avoid_1 : (n_opt_1+1)*nely*nelx float
        Array the includes n_opt_1 solutions which are to be avoided during first part of optimization (avoid[0] is a null vector and not used).
    avoid_2 : (n_opt_"+1)*nely*nelx float
        Array the includes n_opt_2 solutions which are to be avoided during second part of optimization (avoid[0] is a null vector and not used).
    void : nely*nelx boolean
        Enforced void elements in design.
    it_steps : int list
        Number of optimization steps for each part of optimization.
    Iar : float array
        Used for stiffnes matrix assembly.
    cMat : float array
        Used for stiffnes matrix assembly.

    Returns
    -------
    x1 : nely*nelx float
        Density field afte first part of optimization.
    x2 : nely*nelx float
        Density field after second part of optimization.
    c : float
        Compliance of x2.

    '''
    beta=0.05
    epsilon_1=1
    epsilon_2=0.25
    penal=3
    E0=1
    nu=0.3
    max_move=0.25
    x1,_ = TO_SIMP(x0,volfrac,penal,beta,epsilon_1,max_move,E0,nu,Iar,cMat,False,void,avoid_1,it_steps[0],it_steps[0])
    x2,_ = TO_SIMP(x1,volfrac,penal,beta,epsilon_1,max_move,E0,nu,Iar,cMat,False,void,avoid_2,it_steps[1],it_steps[2])
    # Enforce sparse solution
    x2,c = TO_SIMP(x2,volfrac,penal,beta,epsilon_2,max_move,E0,nu,Iar,cMat,True,void,avoid_2,0,10)
    return x1, x2, c

## This part of the generation of samples is performed using parallel processing on a number of cores (size= number of cores, rank= core which this specific instance runs on)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


## Define problem parameters (size, mass constraint and enfroced void elements)
nelx=90
nely=45
volfrac=0.4
void=get_void(nely,nelx)

## Define number of steps for local optimization
it_steps=[25,225,275]


## Create arrays used for stiffness matrix assembly
Iar,cMat=make_Conn_matrix(nelx,nely)


## Generate initial material distribution 
if rank==0:
    x0=np.ones((nely,nelx))*volfrac
else:
    x0=np.random.rand(nely,nelx)**(1.5)
    
## Number of design created per processor core
perrank=100 

## Prepare solution arrays
X_rank=np.zeros((perrank,nely,nelx))
X_sdf_rank=np.zeros((perrank,nely,nelx))
C_rank=np.zeros(perrank)
## Prepare array of previous solutions
Avoid_1_rank=np.zeros((1,nely,nelx))
Avoid_2_rank=np.zeros((1,nely,nelx))


## Wait for every rank
comm.Barrier()

## Generate training designs
for i in range(perrank):
    ## Generate new design
    x1,X_rank[i,:,:],C_rank[i]=get_design(x0,volfrac,Avoid_1_rank,Avoid_2_rank,void,it_steps,Iar,cMat)
    ## Add design to previous solutions
    Avoid_1_rank=np.concatenate((Avoid_1_rank,x1[np.newaxis,:,:]),0)
    Avoid_2_rank=np.concatenate((Avoid_2_rank,X_rank[[i],:,:]),0)
    ## Generate and save signed distance field version of design
    X_sdf_rank[i,:,:]=simp2sdf(X_rank[i,:,:])

    
## Combine solutions from every rank
comm.Barrier()
if rank==0:
    C=np.zeros((size,perrank))
    X=np.zeros((size,perrank,nely,nelx))
    X_sdf=np.zeros((size,perrank,nely,nelx))
    X0=np.zeros((size,nely,nelx))
else:
    C=None
    X=None
    X_sdf=None
    X0=None 
    
comm.Gather(C_rank,C,root=0)
comm.Gather(X_rank,X,root=0)
comm.Gather(X_sdf_rank,X_sdf,root=0)
comm.Gather(x0,X0,root=0)


## Export solutions
if rank==0:
    np.save('Sample_data/X_simp.npy',X)
    np.save('Sample_data/X_sdf.npy',X_sdf)
    np.save('Sample_data/C.npy',C)
    np.save('Sample_data/X0.npy',X0)
    