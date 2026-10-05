"""Resolve conductive inlet heat flux independently of global T indicators."""
import numpy as np


def refine_burner_boundary_grid(z,T,Y,mass_flux,pressure,thermo,transport,*,pe_limit=.02,cells=3):
    """Subdivide inlet intervals to bound their thermal cell Peclet number.

    Global normalized temperature indicators can miss weak inlet gradients.
    This adds nodes to the cold boundary region while retaining all old nodes.
    It is a mesh indicator, not an accuracy guarantee or a change to physics.
    """
    z=np.asarray(z,dtype=float);T=np.asarray(T);Y=np.asarray(Y)
    if (len(z)<4 or z[0]!=0. or np.any(np.diff(z)<=0.) or not np.isfinite(z).all()
            or not np.isfinite(pe_limit) or pe_limit<=0. or mass_flux<=0. or cells<1):
        raise ValueError('Finite increasing grid, positive mass flux and Peclet limit required')
    cells=min(cells,len(z)-1)
    mw=transport.mech.molecular_weights
    cp=float(thermo.cp_mass(T[0],Y[:,0]))
    _,_,conductivity,_=transport.eval_faces_poly_fast(T[:1],pressure,Y[:,:1],1./mw)
    spacing=pe_limit*conductivity[0]/(mass_flux*cp)
    intervals=np.maximum(1,np.ceil(np.diff(z[:cells+1])/spacing*(1.-64.*np.finfo(float).eps)).astype(int))
    if intervals.sum()>6000:
        raise ValueError('Boundary refinement exceeds its node budget')
    points=[np.linspace(z[i],z[i+1],count+1)[:-1] for i,count in enumerate(intervals)]
    return np.r_[np.concatenate(points),z[cells:]]
