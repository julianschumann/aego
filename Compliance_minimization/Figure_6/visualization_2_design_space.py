import numpy as np
from mpi4py import MPI
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

def evaluate(x0,volfrac,void,Iar,cMat):
    beta=0.05
    epsilon_2=0.25
    nelx=90
    nely=45
    penal=3
    E0=1
    nu=0.3
    max_move=0.25
    if np.mean(x0)>volfrac:
        x0=x0*volfrac/np.mean(x0)
        
    _,c1 = TO_SIMP(x0,nelx,nely,volfrac,penal,beta,epsilon_2,max_move,E0,nu,Iar,cMat,True,void,np.zeros((1,nely,nelx)),0,10)
    _,c2 = TO_SIMP(x0,nelx,nely,volfrac,penal,beta,epsilon_2,max_move,E0,nu,Iar,cMat,True,void,np.zeros((1,nely,nelx)),0,0)
    return c1,c2


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nelx=90
nely=45
volfrac=0.4+0.6*0.0001
void=get_void(nely,nelx)

Iar,cMat=make_Conn_matrix(nelx,nely)

num_samples=100000 
perrank=int(np.ceil(num_samples/size))
num_samples=perrank*size

C_rand_rank=np.zeros(perrank)
C_rand_opt_rank=np.zeros(perrank)
C_rand_inner_opt_rank=np.zeros(perrank)
X_rand_rank=np.zeros((perrank,nely,nelx))
for i in range(perrank):
    X_rand_rank[i]=np.random.rand(nely,nelx)**1.5
    X_rand_rank_inner_i=X_rand_rank[i]*0.5+0.2
    
    C_rand_opt_rank[i],C_rand_rank[i]=evaluate(X_rand_rank[i],volfrac,void,Iar,cMat)
    C_rand_inner_opt_rank[i],_=evaluate(X_rand_rank_inner_i,volfrac,void,Iar,cMat)
    
    
if rank==0:
    X_rand=np.zeros((perrank*size,nely,nelx))
    C_rand=np.zeros(perrank*size)
    C_rand_opt=np.zeros(perrank*size)
    C_rand_inner_opt=np.zeros(perrank*size)
else:
    X_rand=None
    C_rand=None
    C_rand_opt=None
    C_rand_inner_opt=None
    
comm.Gather(C_rand_rank,C_rand,root=0)
comm.Gather(C_rand_opt_rank,C_rand_opt,root=0)
comm.Gather(C_rand_inner_opt_rank,C_rand_inner_opt,root=0)
comm.Gather(X_rand_rank,X_rand,root=0)

if rank==0:
    np.save('Sample_data/X_rand.npy',X_rand)
    np.save('Sample_data/C_rand_opt.npy',C_rand_opt)
    np.save('Sample_data/C_rand_inner_opt.npy',C_rand_inner_opt)
    np.save('Sample_data/C_rand.npy',C_rand)
    
    