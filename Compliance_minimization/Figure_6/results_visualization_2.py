import numpy as np


def pdf(C,K,c):
    C=np.sort(C)
    Cp=np.pad(C,K+1,'edge')
    Cp[:K]=2*C[0]-C[-1]-1
    Cp[-K:]=2*C[-1]-C[0]+1
    p=np.zeros(len(c))
    for i in range(len(c)):
        ci=c[i]
        k=np.searchsorted(C,ci)
        Ci=Cp[k+1:k+1+2*K]
        Di=np.abs(Ci-ci)
        R=np.median(np.concatenate((Di,[0])))
        p[i]=K/R
    p=p/(2*len(C))
    p[1:-1]=0.25*(2*p[1:-1]+p[:-2]+p[2:])
    return p
        

C=np.load('Sample_data/C_rand_opt.npy')
C_int=np.load('Sample_data/C_rand_inner_opt.npy')
C_D=np.load('Sample_data/C_D.npy')
C_D_rand=np.load('Sample_data/C_D_rand.npy')
   
c=np.linspace(80,130,201)
p_d=pdf(C_D,5000,c)
p=pdf(C,5000,c)
p_d_rand=pdf(C_D_rand,5000,c)
p_int=pdf(C_int,5000,c)


f=open('probability.txt','w+')
f.write('c pr pri pd pdr \n')
for i in range(len(c)):
    f.write('{:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e}\n'.format(c[i], p[i],p_int[i], p_d[i], p_d_rand[i]))
    
f.close()

    

        