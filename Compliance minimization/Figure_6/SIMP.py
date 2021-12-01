import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
#from scikit.sparse.cholmod import cholesky




def make_Conn_matrix(nelx,nely):
    #returns the pair with all nonzero entries in stiffness matrix
    nEl = nelx * nely #number  of  elements
    nodeNrs = np.transpose(np.arange((1 + nelx) * (1 + nely)).reshape( 1+nelx , 1+nely )).astype(int)
    cVec1 = np.transpose(2 * (nodeNrs[:- 1,:- 1 ]+1) + 1).reshape( nEl , 1 )
    cVec2 = np.array( [[ 0, 1, 2 * nely + 2,2 * nely + 3,2 * nely + 0,2 * nely + 1 , -2,  -1 ]] ).astype(int)
    cMat = cVec1 + cVec2 # connectivity  matrix (in each line are the dof, from lower left corner counterclockwise)
    sI=np.zeros(36)
    sII=np.zeros(36)
    i=0
    for j in range(8):
        sI[i:i+8-j]=np.arange(j,8)
        sII[i:i+8-j] = j
        i=i+8-j   
    iK=np.transpose(cMat[:,sI.astype(int)])
    jK=np.transpose(cMat[:,sII.astype(int)])
    Iar=np.zeros((36*nelx*nely,2))
    Iar[:,0]=np.transpose(np.array(iK)).reshape(36*nelx*nely)
    Iar[:,1]=np.transpose(np.array(jK)).reshape(36*nelx*nely)
    for i in range(36*nelx*nely):
        if Iar[i,0]<Iar[i,1]:
            Iar[i,:]=np.flip(Iar[i,:])
    return (Iar-1).astype(int),(cMat-1).astype(int)

def TO_SIMP(x,x0,penal,beta,epsilon,max_move,E0,nu,Iar,cMat,opt,void,avoid,it_avoid,it_max):
    '''
    This function solves a Messerschmitt-Bölkow-Blohm beam problem with enforced void elements
    and a possible application of deflation.
    
    Based on [1] for the implementation of FEM and SIMP and on [2] for the implementation of delfation.
    
    
    [1]: Erik Andreassen, Anders Clausen, Mattias Schevenels, Boyan S Lazarov, and Ole Sigmund. Efficient topology optimization in matlab using 88 lines of code. Structural and Multidisciplinary Optimization, 43(1):1–16, 2011.
    
    [2]: Ioannis PA Papadopoulos, Patrick E Farrell, and Thomas M Surowiec. Computing multiple solutions of topology optimization problems. SIAM Journal on Scientific Computing, 43(3):A1555–A1582, 2021.
    
    
    Parameters
    ----------
    x : nely*nelx float 
        Densitiy field of the compliance minimization problem
    x0 :  float
        Part of volume that has to be filled.
    penal : float
        Penelization from density to Young's modulus.
    beta : float
        First parameter for Ginzburg-Landau energy term.
    epsilon : float
        Second parameter for Ginzburg-Landau energy term.
    max_move : float
        The maximum absolut deifference for the mass density of every element between iterations.
    E0 : float
        Maximum Young's modulus corresponding to x_i,j=1.
    nu : float
        Poisson ratio.
    Iar : float array
        Used for stiffnes matrix assembly.
    cMat : float array
        Used for stiffnes matrix assembly.
    opt : boolean
        Decides to give out either the best sample of the process (False) or the last(True).
    void : nely*nelx boolean
        True elements have an enforcement of x_i,j=0.
    avoid : (n_opt+1)*nely*nelx float
        Array the includes n_opt solutions which are to be avoided during optimization (avoid[0] is a null vector and not used).
    it_avoid : int
        Number of iterations for which solutions in avoid are used during optimization.
    it_max : int
        Number of iterations (it_max=0 only leads to evaluation). 

    Returns
    -------
    nely*nelx float
        The finial density distribution.
    float
        The compliance for the final solution.

    '''
    nely,nelx=x.shape
    loop = 0
    cmin=1000000
    KE,KE_vector = local_stiffness(nu)
    
    
    # Optimization runs until mass constraint is fullfilled
    if np.mean(x)<=x0*1.003:
        mass_ok=True
    else:
        mass_ok=False
    # START ITERATION
    while loop<it_max+1 or mass_ok==False: 
        # FEM is used to generate displacement field for current density field 
        x[void]=0
        xT=np.transpose(x).reshape(nelx*nely,1)
        U,c=FE(nelx,nely,x,penal,E0,nu,Iar,KE_vector)
        
        # Fullfillment of mass requirement is checked
        if mass_ok==False:
            if np.mean(x)<=x0*1.003:
                xmin=x
                cmin=c
                mass_ok=True
        else:
            if c<cmin:
                xmin=x
                cmin=c
        
        # Density field is updated
        if loop<it_max or mass_ok==False: 
            # Get sensitivity based on compliance
            dx=-penal*E0*xT**(penal-1)
            k=np.transpose(np.array([np.sum(np.matmul(U[cMat,0],KE)*U[cMat,0],1)]))
            dc=dx*k
            dc = np.transpose(dc.reshape(nelx , nely))
            
            # Get sensitivity to Ginzburg-Landau energy term
            xL = np.pad(x,1,'edge')
            DyGL = np.abs(xL[1:,1:-1]-xL[:-1,1:-1])*np.sign(xL[1:,1:-1]-xL[:-1,1:-1])
            DxGL = np.abs(xL[1:-1,1:]-xL[1:-1,:-1])*np.sign(xL[1:-1,1:]-xL[1:-1,:-1])
            dcGL = 1/epsilon*(1-2*x) + epsilon*(DyGL[:-1,:]-DyGL[1:,:]+DxGL[:,:-1]-DxGL[:,1:])
            dc = dc+beta*dcGL
            
            # Get sensitivity to avoid previous solutions using deflation
            if loop<it_avoid:
                # first row of avoid is zeros placeholder
                if len(avoid)>1:
                    dc= dc - 50*np.sum((x[np.newaxis,:,:]-avoid[1:,:,:])/(np.sum((x[np.newaxis,:,:]-avoid[1:,:,:])**2,axis=(1,2))**2)[:,np.newaxis,np.newaxis],0)


            # update density field with combined sensitivity
            xU=x+max_move
            xL=x-max_move
            ocP = np.sqrt(x*np.maximum(-dc,0.0 ))
            L = [ 0, 1000000]
            # Enforce mass requirement
            while (L[1] - L[0]) / (L[1] + L[0]+1e-6) > 1e-4:
                lmid = 0.5 * (L[1] + L[0])
                x = np.maximum( np.maximum( np.minimum( np.minimum( ocP / (lmid+1e-6) , xU ), 1 ), xL ), 1e-4 )
                if np.mean( x ) > x0: 
                    L[0] = lmid 
                else:
                    L[1] = lmid
        loop = loop + 1
        
    if opt:
        return xmin, cmin
    else:
        return x,c


# Helper function required for the FEM evaluation

def Young(E0,den,penal):
    Emin=E0*1e-4
    return Emin + (E0-Emin)*den**penal

def local_stiffness(nu):
    k=np.array([0, 1/2-nu/6,   1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, -1/4+nu/12, -1/8-nu/8,  nu/6, 1/8-3*nu/8])
    KE = 1/(1-nu**2)*np.array([[k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8]],
           [k[2], k[1], k[8], k[7], k[6], k[5], k[4], k[3]],
           [k[3], k[8], k[1], k[6], k[7], k[4], k[5], k[2]],
           [k[4], k[7], k[6], k[1], k[8], k[3], k[2], k[5]],
           [k[5], k[6], k[7], k[8], k[1], k[2], k[3], k[4]],
           [k[6], k[5], k[4], k[3], k[2], k[1], k[8], k[7]],
           [k[7], k[4], k[5], k[2], k[3], k[8], k[1], k[6]],
           [k[8], k[3], k[2], k[5], k[4], k[7], k[6], k[1]]])
    c1 = np.array([12,3, -6, -3, -6, -3,0,3,12,3,0, -3, -6,-3, -6,12, -3,0, -3, -6,3,12,3,-6,3, -6,12,3, -6, -3,12,3,0,12, -3,12])
    c2 = np.array([-4,3, -2,9,2, -3,4, -9, -4, -9,4, -3,2,9, -2, -4, -3,4,9,2,3, -4, -9, -2,3,2, -4,3, -2,9, -4, -9,4, -4, -3, -4])
    V = 1/(1-nu**2)/24*( c1 + nu * c2 )
    KE_vector=np.zeros((36,1))
    KE_vector[:,0]=V
    return KE,KE_vector

def FE(nelx,nely,x,penal,E0,nu,Iar,KE_vector):
    x=np.transpose(x).reshape(nelx*nely,1)
    E=Young(E0,x,penal)
    nDof=2*(nelx+1)*(nely+1)
    Ev = np.transpose(KE_vector * np.transpose(E)).reshape(len(KE_vector) * nelx*nely , 1 );
    K=sp.coo_matrix((Ev[:,0], (Iar[:,0], Iar[:,1])), shape=(nDof, nDof))
    # DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
    U = np.zeros((nDof,1))
    F = sp.csr_matrix((nDof, 1))
    F[1,0] = -1; 
    fixeddofs   = np.ones(nely+2)*(nDof-1)
    fixeddofs[:-1]=np.arange(0,2*(nely+1),2)
    fixeddofs=fixeddofs.astype(int)
    alldofs     = np.arange(nDof)
    freedofs    = np.setdiff1d(alldofs,fixeddofs)

    # I am not sure if the following three lines are the most optimal solution possible
    # Implementiung a form of Cholesky facotorization instead of using spsolve will likely be faster
    K=K+ sp.tril(K,k=-1,format='coo').transpose()
    K=K.tocsr()
    U[freedofs,0] = spsolve(K[freedofs[:, np.newaxis],freedofs], F[freedofs,0])  
    C=np.dot(U[:,0],F.toarray()[:,0])    
    return U,C