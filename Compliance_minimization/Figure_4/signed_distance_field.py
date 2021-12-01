import numpy as np


def sdf2simp(x_sdf):
    '''
    Function allows the transformation of a signed distance field into a density field.

    Parameters
    ----------
    x_sdf : nely*nelx float
        Signed distance field.

    Returns
    -------
    x_simp : nely*nelx float
        Density field.

    '''
    x_simp=np.clip(x_sdf+0.5,0,1)
    return x_simp

def simp2sdf(x_simp):
    '''
    Function allows the transformation of a density field into a quasi signed distance field.
    
    See Appendix A.8 of my master thesis.

    Parameters
    ----------
    x_simp : nely*nelx float
        Density field.

    Returns
    -------
    x_sdf : nely*nelx float
        Signed distance field.

    '''
    
    
    filtera=np.array([[1.41,1,1.41],[1,0,1],[1.41,1,1.41]])
    
    nely=x_simp.shape[0]
    nelx=x_simp.shape[1]
    nel=np.sqrt(nely**2+nelx**2)+2
    x_sdf=x_simp-0.5
    
    
    #Find boundaries
    Indexl=np.array(np.where(x_sdf<=np.min(x_sdf)+0.001))
    Indexu=np.array(np.where(x_sdf>=np.max(x_sdf)-0.001))
    x_sdf[Indexl[0],Indexl[1]]=-nel-1
    x_sdf[Indexu[0],Indexu[1]]=nel+1
    
    # Go from boundaries to center of voids while assiging distance
    while len(Indexl[0])>0:
        x_sdf_prior=np.pad(np.copy(x_sdf),1,'edge')
        for i in range(len(Indexl[0,:])):
            Iy=Indexl[0,i]
            Ix=Indexl[1,i]
            neighbor=np.copy(x_sdf_prior[Iy:Iy+3,Ix:Ix+3])
            neighbor=np.minimum(neighbor,filtera*0.5)
            x_sdf[Iy,Ix]=np.max(neighbor-filtera)
        Indexl=np.array(np.where(x_sdf<=-nel-1))
        
    # Go from boundaries to center of mass while assigning densities
    while len(Indexu[0])>0:
        x_sdf_prior=np.pad(np.copy(x_sdf),1,'edge')
        for i in range(len(Indexu[0,:])):
            Iy=Indexu[0,i]
            Ix=Indexu[1,i]
            neighbor=np.copy(x_sdf_prior[Iy:Iy+3,Ix:Ix+3])
            neighbor=np.maximum(neighbor,-filtera*0.5)
            x_sdf[Iy,Ix]=np.min(neighbor+filtera)
        Indexu=np.array(np.where(x_sdf>=nel+1))                  
    return x_sdf
