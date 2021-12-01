import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model 
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

def cost(X,dim):
    C=np.sum(-X*np.sin(np.sqrt(np.abs(X))),1)+418.9829*X.shape[1]
    return C

def cost_dec(Z,decoder,xmin,xmax,dim):
    X=decoder.predict(Z)
    X=X*max(np.abs(xmin),np.abs(xmax))
    return cost(X,dim)
    

def gradient(X,dim):
    G=-np.sin(np.sqrt(np.abs(X)))-X*np.cos(np.sqrt(np.abs(X)))*0.5/np.sqrt(np.abs(X))*np.sign(X)
    return G

def one_run(num_samples, number_advances,dim,edim,xmin,xmax,X_in):
    X_in=X_in*(xmax-xmin)+xmin
    
    dim1=np.arange(dim)
    dim_vector=np.tile(dim1[np.newaxis,:],(num_samples,1))
    
    C=np.zeros((number_advances+1,num_samples))
    C[0,:]=cost(X_in,dim_vector)
    M=np.zeros((num_samples,dim))
    V=np.zeros((num_samples,dim))
    eta=20
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    
    for i in range(number_advances):
        print('advance step {}'.format(i))
        G=gradient(X_in,dim_vector)
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
        C[i+1,:]=cost(X_in,dim_vector)
    
    C_min_rand=np.min(C,1)
    eval_rand=np.linspace(0,number_advances*num_samples,number_advances+1)
    
    index_train=C[i+1,:]<np.median(C[i+1,:])
    X_train=X[index_train,:]
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
    AE.fit(x=X_train/max(np.abs(xmin),np.abs(xmax)), y=X_train/max(np.abs(xmin),np.abs(xmax)), batch_size=int(num_samples/20), epochs=epoch)
    
    # Optimize in latent space
    F=0.6
    prob_change=0.95
    imax=2000
    train_samples=5*edim    
    dim1=np.arange(dim)
    dim_vector=np.tile(dim1[np.newaxis,:],(train_samples,1))
    Z=np.random.rand(train_samples,edim)
    C=np.zeros((imax+1,train_samples))
    C[0,:]=cost_dec(Z,decoder,xmin,xmax,dim_vector)
    
    for i in range(imax):
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
        Ccom=cost_dec(Zcom,decoder,xmin,xmax,dim_vector)
        Z[C[i,:]>Ccom,:]=np.copy(Zcom[C[i,:]>Ccom,:])
        C[i+1,:]=np.copy(C[i,:])
        C[i+1,C[i,:]>Ccom]=Ccom[C[i,:]>Ccom]
    
    C_min_ae=np.min(C,1)
    eval_ae=eval_rand[-1]+np.linspace(0,train_samples*imax,imax+1)
    
    #post opt
    i_post=2000
    X_post=decoder.predict(Z)*max(np.abs(xmin),np.abs(xmax))
    C_post=np.zeros((i_post+1,train_samples))
    C_post[0,:]=cost(X_post,dim_vector)
    M=np.zeros((train_samples,dim))
    V=np.zeros((train_samples,dim))
    eta=0.5
    betam=0.9
    betav=0.999
    betamh=0.9
    betavh=0.999
    
    for i in range(i_post):
        G=gradient(X_post,dim_vector)
        M=betam*M+(1-betam)*G
        V=betav*V+(1-betav)*G**2
        Mh=M/(1-betamh)
        Vh=V/(1-betavh)
        betamh=betamh*betam
        betavh=betavh*betav
        D=eta*Mh/(Vh**0.5 +1e-8)
        X_post=X_post-D
        X_post=np.clip(X_post,xmin,xmax)
        C_post[i+1,:]=cost(X_post,dim_vector)
    C_min_post=np.min(C_post,1)
    eval_post=eval_ae[-1]+np.linspace(0,train_samples*i_post,i_post+1)
    return C_min_rand, C_min_ae, C_min_post, eval_rand, eval_ae, eval_post

np.random.seed(0)

dim=100
edim=5

num_samples=10000
num_advances=125

xmax=500
xmin=-500

X_in=np.random.rand(num_samples,dim)
    
c_min_rand,c_min_ae,c_min_post,ev_rand,ev_ae,ev_post=one_run(num_samples,num_advances,dim,edim,xmin,xmax,X_in)
        
fpost=open('f_1post.txt','w+')
fpost.write('n c \n')
for k in range(len(c_min_post)):
    fpost.write(' {:10.3e} {:10.3e} \n'.format(ev_post[k],max(min(c_min_post[k],np.min(c_min_post[:k+1])),10e-6)))
fpost.close()

fae=open('f_1ae.txt','w+')
fae.write('n c \n')
for k in range(len(c_min_ae)):
    fae.write(' {:10.3e} {:10.3e} \n'.format(ev_ae[k],max(min(c_min_ae[k],np.min(c_min_ae[:k+1])),10e-6)))
fae.close()
 
frand3=open('f_1rand3.txt','w+')
frand3.write('n c \n')
for k in range(len(c_min_rand)):
    frand3.write('{:10.3e} {:10.3e} \n'.format(ev_rand[k],max(min(c_min_rand[k],np.min(c_min_rand[:k+1])),10e-6)))
frand3.close()