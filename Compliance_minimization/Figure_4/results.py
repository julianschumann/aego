import numpy as np
import os
from SIMP import make_Conn_matrix, TO_SIMP
from PIL import Image


def evaluate_design(X,volfrac,Iar,cMat,void,opt_it):
    beta=0.05
    epsilon_1=1
    epsilon_2=0.25
    nelx=90
    nely=45
    penal=3
    E0=1
    nu=0.3
    max_move=0.25
               
    (n,nely,nelx)=X.shape
    avoid=np.zeros((1,nely,nelx))
    C=np.zeros(n)
    X_out=np.zeros(X.shape)
    for i in range(n):
        X_out[i,:,:], _    = TO_SIMP(X[i,:,:]    , nelx, nely, volfrac, penal, beta, epsilon_1, max_move, E0, nu, Iar, cMat, False, void, avoid, 0, opt_it)
        ## enfroce a sparse density in designs
        X_out[i,:,:], C[i] = TO_SIMP(X_out[i,:,:], nelx, nely, volfrac, penal, beta, epsilon_2, max_move, E0, nu, Iar, cMat, True , void, avoid, 0, 10)
    return X_out,C



def arr_argmin(x):
    return np.unravel_index(np.argmin(x), x.shape)



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




























## Set dimensionality of problem
nely=45
nelx=90
volfrac=0.4
void=get_void(nely, nelx)

## Set up all the possible haperparameters of the optimization process (I know that the could be recover automatically from the result filenames, but I am too lazy to implement that right now)
encoding_dim=np.array([25,100])
NN_typ=['max','sdf']
Opt_it=np.array([1,25])
cases=4



generate_data=True
## Analyze data if results of this are not already available
if generate_data==True: 
    ## Solve the compliance minimization problem using local optimization with an initial homogeneous material destibution
    Iar,cMat=make_Conn_matrix(nelx,nely)
    X_homo,C_homo=evaluate_design(np.ones((1,nely,nelx))*volfrac,volfrac,Iar,cMat,void,1000)
    X_homo=X_homo[0,:,:]
    C_homo=C_homo[0]
    
    ## Load training samples and find the least compliant one
    C_random=np.load('Sample_data/C.npy')
    ind_min=arr_argmin(C_random)
    X_random=np.load('Sample_data/X_simp.npy')
    C_random=C_random[ind_min]
    X_random=X_random[ind_min[0],ind_min[1],:,:]
    
    
    ## Start analyzing the optimization results, by setting up empty containers
    C_post=np.zeros((len(encoding_dim),2,2,2,len(NN_typ),len(Opt_it),cases))   
    X_post=np.zeros((len(encoding_dim),2,2,2,len(NN_typ),len(Opt_it),cases,nely,nelx))   
    Rec_losses=np.zeros((len(encoding_dim),2,2,2,len(NN_typ),cases))
    Rec_val_losses=np.zeros((len(encoding_dim),2,2,2,len(NN_typ),cases))
    
    ## Vary the latent space dimensionality
    for k in range(len(encoding_dim)):
        edim=encoding_dim[k]
        ## Vary the use of pretraining for encoder and decoder
        for l in range(2):
            pre=l
            ## Vary the use of a discriminator network
            for m in range(2):
                dis=m
                ## Vary the use of a surrogate model network
                for n in range(2):            
                    surr=n
                    ## Vary the type of encoding of designs
                    for o in range(len(NN_typ)):
                        typ=NN_typ[o]
                        ## Vary the instace of the optimization run
                        for q in range(cases):
                            example_case=q
                            print('edim={}, pre={}, AAE={}, Surrogate={}, typ='.format(edim,pre,dis,surr)+typ)
                            Rec_losses[k,l,m,n,o,q]=np.load('Sample_data/edim={}_pre={}_AAE={}_surr={}_'.format(edim,pre,dis,surr)+typ+'_case={}_training_losses.npy'.format(example_case))  
                            Rec_val_losses[k,l,m,n,o,q]=np.load('Sample_data/edim={}_pre={}_AAE={}_surr={}_'.format(edim,pre,dis,surr)+typ+'_case={}_validation_losses.npy'.format(example_case))                     
                            ## Vary the number of local optimization steps in the cost function
                            for p in range(len(Opt_it)):
                                opt_it=Opt_it[p]    
                                ## Load the respective results
                                data=np.load('Sample_data/edim={}_pre={}_AAE={}_surr={}_'.format(edim,pre,dis,surr)+typ+'_opt={}_case={}_Results.npy'.format(opt_it,example_case),allow_pickle=True)
                                [X_final,F_final,_]=data
                                ## Save the optimial result
                                C_post[k,l,m,n,o,p,q]=np.min(F_final) 
                                X_post[k,l,m,n,o,p,q]=X_final[np.argmin(F_final),:,:]
                                        
    ## Save the results
    data=np.array([X_post,X_random,X_homo,C_post,C_random,C_homo,Rec_losses,Rec_val_losses])
    np.save('Sample_data/Results_comp_min.npy',data)

else:
    ## If already existent, load the results
    data=np.load('Sample_data/Results_comp_min.npy',allow_pickle=True)
    [X_post,X_random,X_homo,C_post,C_random,C_homo,Rec_losses,Rec_val_losses]=data
    


def save_as_png(arr,name):
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


## Save the best result of different stages of the optimization process


img_h = Image.fromarray(np.uint8((1-X_homo) * 255) , 'L')
arr_h = np.asarray(img_h)
save_as_png(X_homo,'X_homo')

img_hn = Image.fromarray(np.uint8((1-X_random) * 255) , 'L')
arr_hn = np.asarray(img_hn)
save_as_png(X_random,'X_rand')

img_p = Image.fromarray(np.uint8((1-X_post[1,1,1,1,1,1,3]) * 255) , 'L')
arr_p = np.asarray(img_p)
save_as_png(X_post[arr_argmin(C_post)],'X_post')




