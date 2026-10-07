"""Fused counterflow residual assembly; physical properties stay native."""
import numpy as np
from numba import njit


@njit(cache=True)
def assemble_residual(m,V,T,Y,L,rho,cp,omega,hk,lam,J,mu,dz,
                      fuel_Y,oxidizer_Y,mfuel,moxidizer,Tfuel,Toxidizer,
                      mref,aref,Tref,lref,species_scale,energy_scale,excess,
                      transient,oldV,oldT,oldY,dt,fixed,temperature):
    ns,n=Y.shape;nv=ns+3
    residual=np.zeros(n*nv+1)
    for j in range(n):
        offset=j*nv
        if j==0 or j==n-1:
            if j==0:
                residual[offset]=(m[j]-mfuel)/mref
                residual[offset+2]=(T[j]-Tfuel)/Tref
                feed=fuel_Y;face=0
            else:
                residual[offset]=(m[j]+moxidizer)/mref
                residual[offset+2]=(T[j]-Toxidizer)/Tref
                feed=oxidizer_Y;face=n-2
            residual[offset+1]=V[j]/aref
            total=0.
            for k in range(ns):
                residual[offset+3+k]=(m[j]*(Y[k,j]-feed[k])+J[k,face])/mref
                total+=Y[k,j]
            residual[offset+3+excess[j]]=total-1.
            continue
        hm,hp=dz[j-1],dz[j]
        volume=.5*(hm+hp)
        residual[offset]=(m[j]-m[j-1]+hm*(rho[j-1]*V[j-1]+rho[j]*V[j]))/mref
        if m[j]>=0.:
            donor=j-1;spacing=hm;direction=1.
        else:
            donor=j+1;spacing=hp;direction=-1.
        gradV=direction*(V[j]-V[donor])/spacing
        gradT=direction*(T[j]-T[donor])/spacing
        residual[offset+1]=(m[j]*gradV+rho[j]*V[j]**2+L
            -(mu[j]*(V[j+1]-V[j])/hp-mu[j-1]*(V[j]-V[j-1])/hm)/volume)/lref
        enthalpy_diffusion=0.;reaction_enthalpy=0.
        for k in range(ns):
            gradY=direction*(Y[k,j]-Y[k,donor])/spacing
            residual[offset+3+k]=(m[j]*gradY+(J[k,j]-J[k,j-1])/volume-omega[k,j])/species_scale
            if transient:
                residual[offset+3+k]+=rho[j]*(Y[k,j]-oldY[k,j])/(dt*species_scale)
            if not fixed:
                gradH=direction*(hk[k,j]-hk[k,donor])/spacing
                enthalpy_diffusion+=.5*(J[k,j-1]+J[k,j])*gradH
                reaction_enthalpy+=hk[k,j]*omega[k,j]
        if fixed:
            residual[offset+2]=(T[j]-temperature[j])/Tref
        else:
            residual[offset+2]=(m[j]*cp[j]*gradT
                -(lam[j]*(T[j+1]-T[j])/hp-lam[j-1]*(T[j]-T[j-1])/hm)/volume
                +enthalpy_diffusion+reaction_enthalpy)/energy_scale
            if transient:
                residual[offset+2]+=rho[j]*cp[j]*(T[j]-oldT[j])/(dt*energy_scale)
        if transient:
            residual[offset+1]+=rho[j]*(V[j]-oldV[j])/(dt*lref)
    residual[-1]=(m[-1]-m[-2]+dz[-1]*(rho[-2]*V[-2]+rho[-1]*V[-1]))/mref
    return residual
