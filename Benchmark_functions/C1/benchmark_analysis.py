import numpy as np

   
X0=np.load('Results/Local_minima.npy')
  
    

D0_barrier=1-np.max(np.abs(X0),1)
D0_center=np.sqrt(np.sum((X0-0.25)**2,1))
D0b=np.min(D0_barrier)
D0cmin=np.min(D0_center[1:])
D0cmax=np.max(D0_center)

X=np.load('Results/X_rand.npy')      

D_barrier=1-np.max(np.abs(X),1)
D_center=np.sqrt(np.sum((X-0.25)**2,1))
Db=np.min(D_barrier)
Dcmin=np.min(D_center[1:])
Dcmax=np.max(D_center)


D=np.zeros((len(X),len(X0)))
for i in range(len(X)):
    print(i)
    D[i]=np.sqrt(np.sum((X[[i],:]-X0[:,:])**2,1))


I=np.argmin(D,1)
Dmin=np.min(D,1)

Iu=np.unique(np.sort(I))