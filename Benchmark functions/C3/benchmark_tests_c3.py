import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model 
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt

xmax=50
xmin=-50

def cost(X,dim):
    Y=1+(X+1)/4
    U= 100*(np.clip(np.abs(X)-10,0,None))**4
    US= np.sum(U,1)
    C1=10*(np.sin(np.pi*Y[:,0]))**2
    C21=(Y-1)**2
    C22=1+10*(np.sin(np.pi*Y))**2
    
    C2=np.sum(C21[:,:-1]*C22[:,1:],1)
    
    C3=US*np.sum(C21[:,:-1],1)
    C=np.pi/X.shape[1]*(C1+C2+C3)
    return C    

def gradient(X,dim):
    Y=1+(X+1)/4
    U= 100*(np.clip(np.abs(X)-10,0,None))**4
    US= np.sum(U,1)
    C21=(Y-1)**2
    C22=1+10*(np.sin(np.pi*Y))**2
    
    dUSdX=400*(np.clip(np.abs(X)-10,0,None))**3*np.sign(X)
    
    dYdX=0.25
    
    dC1dY=np.zeros(X.shape)
    dC1dY[:,0]=np.pi*20*np.sin(np.pi*Y[:,0])*np.cos(np.pi*Y[:,0])
    dC1dX=dC1dY*dYdX
    
    dC21dY=2*(Y-1)
    dC21dX=dC21dY*dYdX
    
    dC22dY=20*np.pi*np.sin(np.pi*Y)*np.cos(np.pi*Y)
    dC22dX=dC22dY*dYdX
    
    dC2dX=np.zeros(X.shape)
    dC2dX[:,0]=dC21dX[:,0]*C22[:,1]
    dC2dX[:,1:-1]=C21[:,:-2]*dC22dX[:,1:-1]+dC21dX[:,1:-1]*C22[:,2:]
    dC2dX[:,-1]=C21[:,-2]*dC22dX[:,-1]
    
    dC3dX=dUSdX*np.tile(np.sum(C21[:,:-1],1)[:,np.newaxis],(1,X.shape[1]))
    dC3dX[:,:-1]=dC3dX[:,:-1]+np.tile(US[:,np.newaxis],(1,X.shape[1]-1))*dC21dX[:,:-1]

    G=np.pi/X.shape[1]*(dC1dX+dC2dX+dC3dX)
    return G

def cost_dec(Z,decoder,xmin,xmax,dim,opt_it):
    X=decoder.predict(Z)
    X=X*max(np.abs(xmin),np.abs(xmax))
    M=np.zeros_like(X)
    V=np.zeros_like(X)
    eta=0.05
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    for i in range(opt_it):
        G=gradient(X,dim)
        M=betam*M+(1-betam)*G
        V=betav*V+(1-betav)*G**2
        Mh=M/(1-betamh)
        Vh=V/(1-betavh)
        betamh=betamh*betam
        betavh=betavh*betav
        D=eta*Mh/(Vh**0.5 +1e-8)
        X=X-D
        X=np.clip(X,xmin,xmax)    
    return cost(X,dim)


def one_run(num_samples, number_advances,i_max,i_de,i_post,dim,Edim,xmin,xmax,opt_it):
    print('generate training samples')
    X_in=np.random.rand(num_samples,dim)
    X_in=X_in*(xmax-xmin)+xmin
    
    
    C=np.zeros((number_advances+1,num_samples))
    c_min_rand=np.zeros(number_advances+1)
    C[0,:]=cost(X_in,dim)
    c_min_rand[0]=np.min(C[0])
    
    M=np.zeros((num_samples,dim))
    V=np.zeros((num_samples,dim))
    eta=3
    betam=0.5
    betav=0.75
    betamh=0.5
    betavh=0.75
    
    for i in range(number_advances):
        print('generate training samples: step {}'.format(i+1))
        G=gradient(X_in,dim)
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
        C[i+1,:]=cost(X_in,dim)
        c_min_rand[i+1]=min(c_min_rand[i],np.min(C[i+1]))
    
    eval_rand=np.linspace(0,number_advances*num_samples,number_advances+1)

    C_min_ae=np.zeros((len(Edim),i_max+1))
    Eval_ae=np.zeros((len(Edim),i_max+1))
    
    C_min_post=np.zeros((len(Edim),i_post+1))
    Eval_post=np.zeros((len(Edim),i_post+1))
    
    for i_e,edim in enumerate(Edim):
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
        
        # Optimize in latent space
        print('Optimize over latent space (m={})'.format(edim))
        F=0.6
        prob_change=0.95
        train_samples=5*edim   
        Z=np.random.rand(train_samples,edim)
        C=np.zeros((i_max+1,train_samples))
        C[0,:]=cost_dec(Z,decoder,xmin,xmax,dim,opt_it)
        C_min_ae[i_e,0]=np.min(C[0])
        for i in range(i_max):
            print('Optimize over latent space (m={}): step {}'.format(edim,i+1))
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
            Ccom=cost_dec(Zcom,decoder,xmin,xmax,dim,opt_it)
            Z[C[i,:]>Ccom,:]=np.copy(Zcom[C[i,:]>Ccom,:])
            C[i+1,:]=np.copy(C[i,:])
            C[i+1,C[i,:]>Ccom]=Ccom[C[i,:]>Ccom]
            C_min_ae[i_e,i+1]=np.min(C[i+1])
        
        Eval_ae[i_e,:]=eval_rand[-1]+np.linspace(0,train_samples*opt_it*i_max,i_max+1)
        
        #post opt
        print('Post processing (m={})'.format(edim))
        X_post=decoder.predict(Z)*max(np.abs(xmin),np.abs(xmax))
        C=np.zeros((i_post+1,train_samples))
        C[0,:]=cost(X_post,dim)
        C_min_post[i_e,0]=np.min(C[0])
        M=np.zeros((train_samples,dim))
        V=np.zeros((train_samples,dim))
        eta=0.05
        betam=0.9
        betav=0.999
        betamh=0.9
        betavh=0.999
        
        for i in range(i_post):
            print('Post processing (m={}): step {}'.format(edim,i+1))
            G=gradient(X_post,dim)
            M=betam*M+(1-betam)*G
            V=betav*V+(1-betav)*G**2
            Mh=M/(1-betamh)
            Vh=V/(1-betavh)
            betamh=betamh*betam
            betavh=betavh*betav
            D=eta*Mh/(Vh**0.5 +1e-8)
            X_post=X_post-D
            X_post=np.clip(X_post,xmin,xmax)
            C[i+1,:]=cost(X_post,dim)
            C_min_post[i_e,i+1]=min(C_min_post[i_e,i],np.min(C[i+1]))
        Eval_post[i_e,:]=Eval_ae[i_e,-1]+np.linspace(0,train_samples*i_post,i_post+1)
        
    print('Optimize over design space')
    F=0.6
    prob_change=0.95
    train_samples=dim   
    Z=np.random.rand(train_samples,dim)
    C=np.zeros((i_de+1,train_samples))
    C[0,:]=cost(Z*max(np.abs(xmin),np.abs(xmax)),dim)
    
    c_min_de=np.zeros(i_de+1)
    c_min_de[0]=np.min(C[0])
    for i in range(i_de):
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
        Ccom=cost(Zcom*max(np.abs(xmin),np.abs(xmax)),dim)
        Z[C[i,:]>Ccom,:]=np.copy(Zcom[C[i,:]>Ccom,:])
        C[i+1,:]=np.copy(C[i,:])
        C[i+1,C[i,:]>Ccom]=Ccom[C[i,:]>Ccom]
        c_min_de[i+1]=np.min(C[i+1])
    eval_de=np.linspace(0,i_de*train_samples,i_de+1)
    return c_min_rand, C_min_ae, C_min_post, c_min_de, eval_rand, Eval_ae, Eval_post, eval_de

np.random.seed(0)

dim=100
Edim=[2,3,4,5,6,7]

num_samples=5000
num_advances=100
i_max=1000
i_de=10000
i_post=1000
opt_it=5


    
c_min_rand, C_min_ae, C_min_post, c_min_de, eval_rand, Eval_ae, Eval_post, eval_de=one_run(num_samples,num_advances,i_max,i_de,i_post,dim,Edim,xmin,xmax,opt_it)


plt.figure(figsize=(10,10))
plt.plot(eval_rand,c_min_rand,'k')

frand=open('f_3rand.txt','w+')
frand.write('n c \n')
for k in range(len(c_min_rand)):
    frand.write('{:10.3e} {:10.3e} \n'.format(eval_rand[k],max(c_min_rand[k],10e-6)))
frand.close()

for i_e,edim in enumerate(Edim):
    e_frac=(edim-np.min(Edim))/(np.max(Edim)-np.min(Edim))
    plt.plot(Eval_ae[i_e],C_min_ae[i_e],c=(0.1, 0.1+0.8*e_frac, 0.9-0.8*e_frac))
    
    fae=open('f_3ae_{}.txt'.format(edim),'w+')
    fae.write('n c \n')
    for k in range(len(C_min_ae[i_e])):
        fae.write(' {:10.3e} {:10.3e} \n'.format(Eval_ae[i_e,k],max(C_min_ae[i_e,k],10e-6)))
    fae.close()
    
    plt.plot(Eval_post[i_e],C_min_post[i_e],':',c=(0.1, 0.1+0.8*e_frac, 0.9-0.8*e_frac))
    
    fpost=open('f_3post_{}.txt'.format(edim),'w+')
    fpost.write('n c \n')
    for k in range(len(C_min_post[i_e])):
        fpost.write(' {:10.3e} {:10.3e} \n'.format(Eval_post[i_e,k],max(C_min_post[i_e,k],10e-6)))
    fpost.close()
 
    

plt.plot(eval_de,c_min_de,'r')

fde=open('f_3de.txt','w+')
fde.write('n c \n')
for k in range(len(c_min_de)):
    fde.write('{:10.3e} {:10.3e} \n'.format(eval_de[k],max(c_min_de[k],10e-6)))
fde.close()

plt.xlabel('n_f')
plt.ylabel('c')
plt.yscale('log')
plt.show()
