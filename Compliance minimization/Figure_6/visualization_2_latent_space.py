import numpy as np
from mpi4py import MPI
from SIMP import TO_SIMP, make_Conn_matrix
from keras.models import load_model

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
    _,c = TO_SIMP(x0,nelx,nely,volfrac,penal,beta,epsilon_2,max_move,E0,nu,Iar,cMat,True,void,np.zeros((1,nely,nelx)),0,10)
    return  c


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


D_rand=load_model('Sample_data/Decoder_random.h5')
D=load_model('../Figure_4/Sample_data/edim=100_pre=1_AAE=1_surr=1_sdf_case=3_Decoder.h5')


C_rank=np.zeros(perrank)
C_rand_rank=np.zeros(perrank)
Z_rank=np.random.rand(perrank,100)
X_rand_rank=D_rand.predict(Z_rank)
X_rank=np.clip(D.predict(Z_rank)+0.5,0,1)

for i in range(perrank):
    C_rand_rank[i]=evaluate(X_rand_rank[i],volfrac,void,Iar,cMat)
    C_rank[i]=evaluate(X_rank[i],volfrac,void,Iar,cMat)
    
    
if rank==0:
    C=np.zeros(perrank*size)
    C_rand=np.zeros(perrank*size)
else:
    C=None
    C_rand=None
    
comm.Gather(C_rand_rank,C_rand,root=0)
comm.Gather(C_rank,C,root=0)

if rank==0:
    np.save('Sample_data/C_D.npy',C)
    np.save('Sample_data/C_D_rand.npy',C_rand)
    
    