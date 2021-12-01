import numpy as np
import os
from PIL import Image

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


## Result of Bayesian optimization is loaded
data=np.load('Sample_data/BO_test.npy',allow_pickle=True)
[X_BO,C_BO,time_needed]=data 

## The resulting design is saved
save_as_png(X_BO,'BO_result')



