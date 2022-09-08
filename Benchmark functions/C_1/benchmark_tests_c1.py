import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model 
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

from os import path

import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
xmax=1
xmin=-1

def mapping(num_loc_minima,dim,R):
    max_distance=0
    edim=5
    Z0=np.random.rand(num_loc_minima,edim)*2-1
    Z0[0,:]=0.1
    X0=np.zeros((num_loc_minima,dim))
    X0[:,:edim]=Z0
    W = special_ortho_group.rvs(dim)
    T0=np.dot(X0,W)
    X0=np.tanh(T0)
    X0=0.9*X0/np.max(np.abs(X0),0)[np.newaxis,:]
    X0=X0+0.1*(1-X0**2)
    checkd=np.min(np.sqrt(np.sum((X0[1:]-X0[[0],:])**2,1)))
    max_distance=max(max_distance,checkd)
    print('min  distance to GO: {:10.3f}>{:1.3f}'.format( checkd,R ))
    print('mean distance to GO: {:10.3f}'.format( np.mean(np.sqrt(np.sum((X0[1:]-X0[[0],:])**2,1))) ))
    print('distance of mean to GO: {:10.3f}>{:1.3f}'.format( np.sqrt(np.sum((np.mean(X0,0)-X0[[0],:])**2)),R ))
    return X0

def cost(X,X0,C0,R):
    n=len(X)
    max_size=200
    if n>max_size:
        nb=int(np.ceil(n/max_size))
        C=np.zeros(n)
        for i in range(nb):
            C[i*max_size:(i+1)*max_size]=Cost(X[i*max_size:(i+1)*max_size],X0,C0,R)
        return C
    else:
        return Cost(X,X0,C0,R)

def gradient(X,X0,C0,R):
    n=len(X)
    max_size=200
    if n>max_size:
        nb=int(np.ceil(n/max_size))
        G=np.zeros((n,X.shape[1]))
        for i in range(nb):
            G[i*max_size:(i+1)*max_size]=Gradient(X[i*max_size:(i+1)*max_size],X0,C0,R)
        return G
    else:
        return Gradient(X,X0,C0,R)
    
  
    

def Cost(X,X0,C0,R):
    Xtf=tf.constant(X)
    X0tf=tf.constant(X0[1:,:])
    Delta=tf.subtract(Xtf[:,tf.newaxis,:],X0tf[tf.newaxis,:,:])
    Dsq=tf.multiply(Delta,Delta)
    D=tf.reduce_sum(Dsq, 2) 
    S1=1+C0[np.newaxis,:]/D
    S2=1/D
    SM1=np.sum(S1,1)
    SM2=np.sum(S2,1)
    C1=SM1/SM2
    C1[np.min(D,1)<1e-7]=C0[np.argmin(D[np.min(D,1)<1e-7],1)]
    
    C2= np.minimum(1,1/R**2*np.sum((X-X0[[0],:])**2,1))
    
    C3=np.min(D,1)
    
    C=(C1+5*C3)*C2
    return C
 

def Gradient(X,X0,C0,R):
    Xtf=tf.constant(X)
    X0tf=tf.constant(X0[1:,:])
    Delta=tf.subtract(Xtf[:,tf.newaxis,:],X0tf[tf.newaxis,:,:])
    Dsq=tf.multiply(Delta,Delta)
    D=tf.reduce_sum(Dsq, 2)  
    S1=1+C0[np.newaxis,:]/D
    S2=1/D
    SM1=np.sum(S1,1)
    SM2=np.sum(S2,1)
    C1=SM1/SM2
    C1[np.min(D,1)<1e-7]=C0[np.argmin(D[np.min(D,1)<1e-7],1)]
    
    C2= np.minimum(1,1/R**2*np.sum((X-X0[[0],:])**2,1))
    
    C3=5*np.min(D,1)
    
    dDdX=2*Delta
    
    dS1dD=-C0[np.newaxis,:]/D**2
    dS1dX=dS1dD[:,:,np.newaxis]*dDdX
    dSM1dX=np.sum(dS1dX,1)
    
    dS2dD=-1/D**2    
    dS2dX=dS2dD[:,:,np.newaxis]*dDdX
    dSM2dX=np.sum(dS2dX,1)
    
    dC1dSM1=1/SM2
    dC1dSM2=-SM1/SM2**2
    dC1dX=dC1dSM1[:,np.newaxis]*dSM1dX+dC1dSM2[:,np.newaxis]*dSM2dX
    dC1dX[np.min(D,1)<1e-7]=0
    
    dC2dX=np.zeros_like(X)
    dC2dX[C2<1,:]=2/R**2*(X[C2<1,:]-X0[[0],:])
    
    dC3dX=10*(X-X0[np.argmin(D,1)+1,:])
    dCdX=(dC1dX+dC3dX)*C2[:,np.newaxis]+(C1[:,np.newaxis]+C3[:,np.newaxis])*dC2dX
    return dCdX

def cost_dec(Z,decoder,xmin,xmax,X0,C0,R,opt_it):
    X=decoder.predict(Z).astype('float64')
    X=X*max(np.abs(xmin),np.abs(xmax))
    M=np.zeros_like(X)
    V=np.zeros_like(X)
    eta=0.01
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    for i in range(opt_it):
        G=gradient(X,X0,C0,R)
        M=betam*M+(1-betam)*G
        V=betav*V+(1-betav)*G**2
        Mh=M/(1-betamh)
        Vh=V/(1-betavh)
        betamh=betamh*betam
        betavh=betavh*betav
        D=eta*Mh/(Vh**0.5 +1e-8)
        X=X-D
        X=np.clip(X,xmin,xmax)    
    return cost(X,X0,C0,R)


def one_run(num_samples, number_advances,i_max,i_de,i_post,dim,Edim,xmin,xmax,opt_it,X0,C0,R,redo_samples):
    print('generate training samples')
    
    if redo_samples==True or path.exists('X1.npy')==False: 
        X_in=np.random.rand(num_samples,dim)
        X_in=X_in*(xmax-xmin)+xmin
        X_in[0,:]=np.mean(X0,0)
        
        C=np.zeros((number_advances+1,num_samples))
        c_min_rand=np.ones(number_advances+1)
        C[0,:]=cost(X_in,X0,C0,R)
        c_min_rand[0]=np.min(C[0])
        
        M=np.zeros((num_samples,dim))
        V=np.zeros((num_samples,dim))
        eta=0.02
        betam=0.5
        betav=0.75
        betamh=0.5
        betavh=0.75
        
        for i in range(number_advances):
            print('generate training samples: step {}, c_min={:1.3f}'.format(i+1,c_min_rand[i]))
            G=gradient(X_in,X0,C0,R)
            M=betam*M+(1-betam)*G
            V=betav*V+(1-betav)*G**2
            Mh=M/(1-betamh)
            Vh=V/(1-betavh)
            betamh=betamh*betam
            betavh=betavh*betav
            D=eta*Mh/(Vh**0.5 +1e-8)
            X=X_in-D
            X=np.clip(X,xmin,xmax)
            X_in=np.copy(X)
            C[i+1,:]=cost(X_in,X0,C0,R)
            c_min_rand[i+1]=min(c_min_rand[i],np.min(C[i+1]))
            print('    min  distance to GO: {:10.3f}'.format( np.min(np.sqrt(np.sum((X-X0[[0],:])**2,1))) ))
        
        Delta=X[:,np.newaxis,:]-X0[np.newaxis,:,:]
        D=np.sum(Delta**2,2)
        
        eval_rand=np.linspace(0,number_advances*num_samples,number_advances+1)
        
        data=[X,eval_rand,c_min_rand,0]
        np.save('X1.npy',np.array(data))
    else:
        data=np.load('X1.npy',allow_pickle=True)
        [X,eval_rand,c_min_rand,_]=data
    
    C_min_ae=np.zeros((len(Edim),i_max+1))
    Eval_ae=np.zeros((len(Edim),i_max+1))
    
    C_min_post=np.zeros((len(Edim),i_post+1))
    Eval_post=np.zeros((len(Edim),i_post+1))
    
    AE_loss=np.zeros(len(Edim))
    
    for i_e,edim in enumerate(Edim):
        tf.random.set_seed(1)
        print('')
        print('Train autoencoder (m={})'.format(edim))
        ## Now train network
        design_space=Input((dim))
        enc=Dense(dim,activation='tanh')(design_space)
        enc=LeakyReLU(alpha=0.25)(enc)
        enc=Dense(dim,activation='tanh')(enc)
        enc=LeakyReLU(alpha=0.25)(enc)
        enc=Dense(edim,activation='sigmoid')(enc)
        encoder=Model(design_space,enc)
        
        latent_space=Input((edim))
        dec=Dense(dim,activation='tanh')(latent_space)
        dec=LeakyReLU(alpha=0.25)(dec)
        dec=Dense(dim,activation='tanh')(dec)
        dec=LeakyReLU(alpha=0.25)(dec)
        dec=Dense(dim,activation='tanh')(dec)
        decoder=Model(latent_space,dec)
        
        AE=Model(design_space,decoder(encoder(design_space)))
        AE.compile('Adam','mse')
        epoch=100
        AE.fit(x=X/max(np.abs(xmin),np.abs(xmax)), y=X/max(np.abs(xmin),np.abs(xmax)), batch_size=int(num_samples/20), epochs=epoch)
        AE_loss[i_e]=np.mean((AE.predict(X)-X)**2)
        
        test_case=cost_dec(encoder.predict(X0).astype('float64'),decoder,xmin,xmax,X0,C0,R,opt_it)
        D_test=np.sqrt(np.sum((AE.predict(X0)-X0)**2,1))
        if np.min(test_case)<1:
            print('global minimum in decoded latent space (error G0: {:1.3f})'.format(D_test[0]))
        else:
            print('global minimum LIKELY NOT in decoded latent space (error G0: {:1.3f})'.format(D_test[0]))
        # Optimize in latent space
        
        np.random.seed(25)
        print('Optimize over latent space (m={})'.format(edim))
        F=0.6
        prob_change=0.95
        train_samples=5*edim   
        Z=np.random.rand(train_samples,edim)
        C=np.zeros((i_max+1,train_samples))
        C[0,:]=cost_dec(Z,decoder,xmin,xmax,X0,C0,R,opt_it)
        C_min_ae[i_e,0]=np.min(C[0])
        for i in range(i_max):
            if np.mod(i,50)==0:
                print('Optimize over latent space (m={}): step {}, c_min={:1.3f}'.format(edim,i+1,C_min_ae[i_e,i]))
            test_case=np.floor(np.random.rand(train_samples,3)*(train_samples-1e-7)).astype('int')
            Za=np.copy(Z[test_case[:,0],:])
            Zb=np.copy(Z[test_case[:,1],:])
            Zc=np.copy(Z[test_case[:,2],:])
            Zcom=Za+F*(Zb-Zc)
            prob=np.random.rand(train_samples,edim)
            Zcom[prob>prob_change]=np.copy(Z[prob>prob_change])
            Zcom[Zcom<=-0.5]=np.random.rand(Zcom[Zcom<=-0.5].shape[0])
            Zcom[Zcom>=1.5]=np.random.rand(Zcom[Zcom>=1.5].shape[0])
            Zcom[Zcom<0]=0
            Zcom[Zcom>1]=1  
            Ccom=cost_dec(Zcom,decoder,xmin,xmax,X0,C0,R,opt_it)
            Z[C[i,:]>Ccom,:]=np.copy(Zcom[C[i,:]>Ccom,:])
            C[i+1,:]=np.copy(C[i,:])
            C[i+1,C[i,:]>Ccom]=Ccom[C[i,:]>Ccom]
            C_min_ae[i_e,i+1]=np.min(C[i+1])
        
        Eval_ae[i_e,:]=eval_rand[-1]+np.linspace(0,train_samples*opt_it*i_max,i_max+1)
        
        #post opt
        print('Post processing (m={})'.format(edim))
        X_post=decoder.predict(Z).astype('float64')*max(np.abs(xmin),np.abs(xmax))
        C=np.zeros((i_post+1,train_samples))
        C[0,:]=cost(X_post,X0,C0,R)
        C_min_post[i_e,0]=np.min(C[0])
        M=np.zeros((train_samples,dim))
        V=np.zeros((train_samples,dim))
        eta=0.001
        betam=0.9
        betav=0.999
        betamh=0.9
        betavh=0.999
        
        for i in range(i_post):
            if np.mod(i,50)==0:
                print('Post processing (m={}): step {}, c_min={:1.3f}'.format(edim,i+1,C_min_post[i_e,i]))
            G=gradient(X_post,X0,C0,R)
            M=betam*M+(1-betam)*G
            V=betav*V+(1-betav)*G**2
            Mh=M/(1-betamh)
            Vh=V/(1-betavh)
            betamh=betamh*betam
            betavh=betavh*betav
            D=eta*Mh/(Vh**0.5 +1e-8)
            X_post=X_post-D
            X_post=np.clip(X_post,xmin,xmax)
            C[i+1,:]=cost(X_post,X0,C0,R)
            C_min_post[i_e,i+1]=min(C_min_post[i_e,i],np.min(C[i+1]))
        Eval_post[i_e,:]=Eval_ae[i_e,-1]+np.linspace(0,train_samples*i_post,i_post+1)
        
        
    print('Optimize over design space')
    F=0.6
    prob_change=0.95
    train_samples=dim   
    Z=np.random.rand(train_samples,dim)
    C=np.zeros((i_de+1,train_samples))
    C[0,:]=cost(Z*max(np.abs(xmin),np.abs(xmax)),X0,C0,R)
    
    c_min_de=np.zeros(i_de+1)
    c_min_de[0]=np.min(C[0])
    for i in range(i_de):
        if np.mod(i,500)==0:
            print('Optimize over design space: step {}'.format(i+1))
        test_case=np.floor(np.random.rand(train_samples,3)*(train_samples-1e-7)).astype('int')
        Za=np.copy(Z[test_case[:,0],:])
        Zb=np.copy(Z[test_case[:,1],:])
        Zc=np.copy(Z[test_case[:,2],:])
        Zcom=Za+F*(Zb-Zc)
        prob=np.random.rand(train_samples,dim)
        Zcom[prob>prob_change]=np.copy(Z[prob>prob_change])
        Zcom[Zcom<=-0.5]=np.random.rand(Zcom[Zcom<=-0.5].shape[0])
        Zcom[Zcom>=1.5]=np.random.rand(Zcom[Zcom>=1.5].shape[0])
        Zcom[Zcom<0]=0
        Zcom[Zcom>1]=1  
        Ccom=cost(Zcom*max(np.abs(xmin),np.abs(xmax)),X0,C0,R)
        Z[C[i,:]>Ccom,:]=np.copy(Zcom[C[i,:]>Ccom,:])
        C[i+1,:]=np.copy(C[i,:])
        C[i+1,C[i,:]>Ccom]=Ccom[C[i,:]>Ccom]
        c_min_de[i+1]=np.min(C[i+1])
    eval_de=np.linspace(0,i_de*train_samples,i_de+1)
    return c_min_rand, C_min_ae, C_min_post, c_min_de, eval_rand, Eval_ae, Eval_post, eval_de, AE_loss

np.random.seed(0)

dim=100
Edim=[2,3,4,5,6,7]


num_loc_minima=1000
R=0.5

num_samples=5000
num_advances=100

i_max=1000
i_post=1000

i_de=10000
opt_it=5

redo_samples=False


X0=mapping(num_loc_minima,dim,R)
C0=np.random.rand(num_loc_minima-1)+1
c_min_rand, C_min_ae, C_min_post, c_min_de, eval_rand, Eval_ae, Eval_post, eval_de, AE_loss=one_run(num_samples,num_advances,i_max,i_de,i_post,dim,Edim,xmin,xmax,opt_it,X0,C0,R,redo_samples)


plt.figure(figsize=(10,10))
plt.plot(eval_rand,c_min_rand,'k')

frand=open('f_1rand.txt','w+')
frand.write('n c \n')
for k in range(len(c_min_rand)):
    frand.write('{:10.3e} {:10.3e} \n'.format(eval_rand[k],max(c_min_rand[k],10e-6)))
frand.close()

for i_e,edim in enumerate(Edim):
    e_frac=(edim-np.min(Edim))/(np.max(Edim)-np.min(Edim))
    plt.plot(Eval_ae[i_e],C_min_ae[i_e],c=(0.1, 0.1+0.8*e_frac, 0.9-0.8*e_frac))
    
    fae=open('f_1ae_{}.txt'.format(edim),'w+')
    fae.write('n c \n')
    for k in range(len(C_min_ae[i_e])):
        fae.write(' {:10.3e} {:10.3e} \n'.format(Eval_ae[i_e,k],max(C_min_ae[i_e,k],10e-6)))
    fae.close()
    
    plt.plot(Eval_post[i_e],C_min_post[i_e],':',c=(0.1, 0.1+0.8*e_frac, 0.9-0.8*e_frac))
    
    fpost=open('f_1post_{}.txt'.format(edim),'w+')
    fpost.write('n c \n')
    for k in range(len(C_min_post[i_e])):
        fpost.write(' {:10.3e} {:10.3e} \n'.format(Eval_post[i_e,k],max(C_min_post[i_e,k],10e-6)))
    fpost.close()
 
    

plt.plot(eval_de,c_min_de,'r')

fde=open('f_1de.txt','w+')
fde.write('n c \n')
for k in range(len(c_min_de)):
    fde.write('{:10.3e} {:10.3e} \n'.format(eval_de[k],max(c_min_de[k],10e-6)))
fde.close()

plt.xlabel('n_f')
plt.ylabel('c')
plt.yscale('log')
plt.ylim([1e-6, 1e2])
plt.show()


plt.figure()
plt.scatter(Edim,AE_loss)
plt.xlabel('m')
plt.ylabel('reconstruction error (mse)')
plt.yscale('log')
plt.show()

ferror=open('f_1error.txt','w+')
ferror.write('m e \n')
for k in range(len(Edim)):
    ferror.write('{} {:10.3e} \n'.format(Edim[k],(AE_loss[k])))
ferror.close()