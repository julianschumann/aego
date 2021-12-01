import numpy as np
from mpi4py import MPI
from SIMP import TO_SIMP, make_Conn_matrix
from PIL import Image


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
    x,c = TO_SIMP(x0,nelx,nely,volfrac,penal,beta,epsilon_2,max_move,E0,nu,Iar,cMat,False,void,np.zeros((1,nely,nelx)),0,0)
    return c


def save_as_gs(arr,name):
    (nely,nelx)=arr.shape
    x_new=1800
    y_new=900
    a_out=np.zeros((y_new,x_new))
    for i in range(y_new):
        for j in range(x_new):
            a_out[i,j]=arr[int(i*nely/y_new),int(j*nelx/x_new)]    
    img = Image.fromarray(np.uint8((1-a_out) * 255) , 'L')
    img.save(name+'.png',format='png')
    return

def save_as_rgb(arr,name):
    (nely,nelx)=arr.shape
    x_new=1800
    y_new=900
    a_out=np.zeros((y_new,x_new))
    for i in range(y_new):
        for j in range(x_new):
            a_out[i,j]=arr[int(i*nely/y_new),int(j*nelx/x_new)]
    rgb=np.zeros((y_new,x_new,3))
    rgb[a_out==0]=[255,255,255]
    rgb[a_out<0]=[0,0,200]
    rgb[a_out>0]=[100,250,0]
    img = Image.fromarray(np.uint8(rgb))
    img.save(name+'.png',format='png')
    return


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nelx=90
nely=45
volfrac=0.4
void=get_void(nely,nelx)

Iar,cMat=make_Conn_matrix(nelx,nely)



X_simp=np.load('../Figure_4/Sample_data/X_simp.npy')
X_simp=X_simp.reshape((X_simp.shape[0]*X_simp.shape[1],X_simp.shape[2],X_simp.shape[3]))

X_simp[X_simp<0.5]=0
X_simp[X_simp>=0.5]=1

mean_ok=np.where(np.mean(X_simp,(1,2))==0.4)[0]
X_simp=X_simp[mean_ok]

C=np.load('../Figure_4/Sample_data/C.npy')
C=C.reshape((C.shape[0]*C.shape[1]))
C=C[mean_ok]


X1=X_simp[np.argmin(C)]
X2=X_simp[np.argmax(C)]

if rank==0:
    save_as_gs(X1,'vis_1_X1')

dX=X2-X1

plus=np.array(np.where(dX>0)).T
minus=np.array(np.where(dX<0)).T

Ip=np.arange(len(plus))
Im=np.arange(len(minus))
np.random.shuffle(Ip)
np.random.shuffle(Im)

dX1=np.zeros_like(X1)
dX2=np.zeros_like(X2)

dX1[plus[Ip[:int(0.075*len(plus))],0],plus[Ip[:int(0.075*len(plus))],1]]=1
dX1[minus[Im[:int(0.075*len(plus))],0],minus[Im[:int(0.075*len(plus))],1]]=-1

dX2[plus[Ip[int(0.075*len(plus)):],0],plus[Ip[int(0.075*len(plus)):],1]]=1
dX2[minus[Im[int(0.075*len(plus)):],0],minus[Im[int(0.075*len(plus)):],1]]=-1


grid_size=101 

perrank=int(np.ceil(grid_size**2/size))
    
comm.Barrier()
dX1=comm.bcast(dX1,root=0)
dX2=comm.bcast(dX2,root=0)

A=np.repeat(np.arange(grid_size),grid_size)/100
B=np.tile(np.arange(grid_size),grid_size)/100

A_rank=A[rank*perrank:min((rank+1)*perrank,grid_size**2)]
B_rank=B[rank*perrank:min((rank+1)*perrank,grid_size**2)]

C_rank=np.zeros(perrank)
for i in range(len(A_rank)):
    X_test=X1+dX1*A_rank[i]+dX2*B_rank[i]
    C_rank[i]=evaluate(X_test,0.4,void,Iar,cMat)
    
    
if rank==0:
    C=np.zeros(perrank*size)
else:
    C=None
    
comm.Gather(C_rank,C,root=0)

if rank==0:
    data=np.zeros((grid_size**2,3))
    data[:,0]=A
    data[:,1]=B
    data[:,2]=C[:grid_size**2]
    np.save('2d_projection.npy',data)
    np.save('vis_1_dX1.npy',dX1)
    np.save('vis_1_dX2.npy',dX2)
    save_as_rgb(np.load('vis_1_dX1.npy'),'vis_1_dX1')
    save_as_rgb(np.load('vis_1_dX2.npy'),'vis_1_dX2')