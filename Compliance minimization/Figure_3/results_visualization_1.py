import numpy as np

data=np.load('2d_projection.npy')

f=open('2d_projection.txt','w+')
f.write('a b c \n')
for i in range(len(data)):
    f.write('{:10.3e} {:10.3e} {:10.3e} \n'.format(data[i,0],data[i,1],data[i,2]))
    if data[i,1]==1 and i<len(data)-1:
        f.write('\n')
f.close()

A=data[:,0].reshape(101,101)
B=data[:,1].reshape(101,101)
Z=data[:,2].reshape(101,101)

n=101

f=open('2d_mesh.txt','w+')
f.write('a b c \n')
for i in range(n):
    if np.mod(i,5)==0:
        for j in range(n):
            f.write('{:10.3f} {:10.3f} {:10.3f}\n'.format(A[j,i],B[j,i],Z[j,i]))
        f.write('\n')
for j in range(n):
    if np.mod(j,5)==0:
        for i in range(n):
            f.write('{:10.3f} {:10.3f} {:10.3f}\n'.format(A[j,i],B[j,i],Z[j,i]))
        if j<n-1:
            f.write('\n')
f.close()

f=open('2d_mesh_2.txt','w+')
f.write('a b c \n')
for i in range(n):
    if np.mod(i,n-1)==0:
        for j in range(n):
            f.write('{:10.3f} {:10.3f} {:10.3f}\n'.format(A[j,i],B[j,i],Z[j,i]))
        f.write('\n')
for j in range(n):
    if np.mod(j,n-1)==0:
        for i in range(n):
            f.write('{:10.3f} {:10.3f} {:10.3f}\n'.format(A[j,i],B[j,i],Z[j,i]))
        if j<n-1:
            f.write('\n')
f.close()

    