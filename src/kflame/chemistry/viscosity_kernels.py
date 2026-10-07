"""Fused native Wilke viscosity without (species,species,faces) temporaries."""
import math
import numpy as np
from numba import njit,prange


@njit(cache=True,parallel=True)
def viscosity_grid(T,X,coeff,quarter_ratio,denominator):
    result=np.empty(len(T))
    ns=X.shape[0]
    for face in prange(len(T)):
        logT=math.log(T[face])
        root=np.empty(ns)
        for k in range(ns):
            c=coeff[k]
            poly=c[0]+logT*(c[1]+logT*(c[2]+logT*(c[3]+logT*c[4])))
            root[k]=abs(T[face]**.25*poly)
        value=0.
        for k in range(ns):
            total=0.
            for j in range(ns):
                ratio=root[k]/max(root[j],1e-150)
                phi=(1.+ratio*quarter_ratio[k,j])**2/denominator[k,j]
                total+=X[j,face]*phi
            value+=max(X[k,face],1e-300)*root[k]**2/max(total,1e-300)
        result[face]=value
    return result
