"""Export steady Jacobians after selected diagnostic replays; never benchmark these times.

Cantera solves with its original banded solver, then does a separate fixed-grid
sparse Newton polish to expose a Jacobian. KFLAME rebuilds its own
production linearization at its final solution. Each matrix has its own variables,
mesh and equation scaling: condition estimates are not a ranking of the solvers.
"""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sweeps',ROOT/'benchmarks/benchmark_flame_sweeps.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def sparse_blocks(j):
    import numpy as np
    from scipy.sparse import bsr_matrix
    data=[];indices=[];ptr=[0]
    for i in range(j.n_blocks):
        if i:data.append(j.lower[i-1]);indices.append(i-1)
        data.append(j.diag[i]);indices.append(i)
        if i+1<j.n_blocks:data.append(j.upper[i]);indices.append(i+1)
        ptr.append(len(indices))
    a=bsr_matrix((np.asarray(data),indices,ptr),shape=j.shape).tocsc()
    a.eliminate_zeros()
    return a


def estimate(a):
    import numpy as np
    from scipy.sparse.linalg import splu,LinearOperator,onenormest,norm
    lu=splu(a)
    inv=LinearOperator(a.shape,matvec=lu.solve,rmatvec=lambda x:lu.solve(x,trans='T'),
                       matmat=lu.solve,rmatmat=lambda x:lu.solve(x,trans='T'),dtype=float)
    np.random.seed(20260927)
    value=float(norm(a,1)*onenormest(inv,t=2,itmax=5))
    rhs=np.ones(a.shape[0]);x=lu.solve(rhs)
    backward=float(np.linalg.norm(a@x-rhs,np.inf)/(norm(a,np.inf)*np.linalg.norm(x,np.inf)+1))
    return dict(kappa1_lower_estimate=value,linear_backward_error=backward)


def worker(root,out,c,backend):
    for k,v in runner.THREADS.items():os.environ[k]=v
    import numpy as np
    from scipy.sparse import csc_matrix,diags,save_npz
    captured={}
    if backend=='native':
        import kflame.flame.solver as sol
        old=sol.solve_free_flame
        depth=0
        def wrap(problem,*a,**kw):
            nonlocal depth
            depth+=1
            try:result=old(problem,*a,**kw)
            finally:depth-=1
            if depth==0:
                captured.update(problem=problem,state=result[0].copy())
            return result
        sol.solve_free_flame=wrap
    else:
        import cantera as ct
        old=ct.FreeFlame.solve
        def wrap(f,*a,**kw):
            result=old(f,*a,**kw);captured['flame']=f;return result
        ct.FreeFlame.solve=wrap
    manifest=runner.read(root/'manifest.json')
    fields,result=runner.solve(c,runner.settings(3,max_seconds=manifest['max_seconds']),backend,
                               root/'inputs'/c['mechanism'],backend)
    folder=out/c['id']/backend;folder.mkdir(parents=True,exist_ok=True)
    meta=dict(case_id=c['id'],backend=backend,accepted=result['accepted'],nodes=len(fields['z']),
              Su=float(fields['u'][0]),diagnostic_only=True,condition=c,
              definition='Steady final-state production linearization, before factorization; no transient shift.',
              estimator='||J||_1 * onenormest(J^-1), t=2, itmax=5, seed=20260927; approximate lower estimate.',
              scaling='A=Dr*J*Dc; Dr=1/max_abs_row(J), Dc=1/max_abs_col(Dr*J). Zero scales become 1.',
              state_scope='Own solver final mesh/variables/equations; magnitudes are not directly comparable between solvers.')
    np.savez_compressed(folder/'state.npz',**fields)
    if not result['accepted']:raise RuntimeError('Conditioning replay did not converge')
    if backend=='native':
        p=captured['problem'];x=captured['state']
        p._current_rdt=0.
        j,_=sol.build_jacobian_steady(lambda v:sol.residual(v,p,rdt=0.),x,p)
        a=sparse_blocks(j)
        meta['extraction']='Fresh KFLAME steady Jacobian on the converged native state, same production Jacobian options.'
    else:
        f=captured['flame'];before=f.to_array();u_before=f.velocity.copy()
        ct.use_sparse(True)
        f.linear_solver=ct.EigenSparseDirectJacobian()
        f.set_max_jac_age(0,0)
        f.solve(loglevel=0,auto=False,refine_grid=False)
        a=csc_matrix(f.linear_solver.jacobian)
        after=f.to_array()
        meta['polish_changes']=dict(max_T_K=float(np.max(abs(before.T-after.T))),
            max_Y=float(np.max(abs(before.Y-after.Y))),Su_percent=float(100*abs(f.velocity[0]-u_before[0])/abs(u_before[0])))
        meta['extraction']='Original banded primal solve followed by separate fixed-grid sparse Newton polish; last assembled Jacobian, not necessarily evaluated at final polished state. Original campaign timing is unaffected.'
    if a.shape[0]==0:raise RuntimeError('Empty Jacobian')
    a.eliminate_zeros();save_npz(folder/'jacobian.npz',a)
    r=np.asarray(abs(a).max(axis=1).toarray()).ravel();r=np.where(r>0,1/np.maximum(r,1e-300),1)
    b=diags(r)@a
    col=np.asarray(abs(b).max(axis=0).toarray()).ravel();col=np.where(col>0,1/np.maximum(col,1e-300),1)
    b=(b@diags(col)).tocsc()
    np.savez_compressed(folder/'scales.npz',row=r,column=col)
    meta.update(dimension=a.shape[0],nnz=a.nnz,raw=estimate(a),equilibrated=estimate(b),
                matrix_sha256=runner.digest(folder/'jacobian.npz'))
    runner.atomic_json(folder/'conditioning.json',meta)
    print(c['id'],backend,meta['raw'],meta['equilibrated'],flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path,
                    default=Path('TESIS_RESUL/corridas/llamas_individuales/thesis_flames_L3'))
    ap.add_argument('--output', type=Path,
                    default=Path('TESIS_RESUL/corridas/reproduccion/thesis_flames_L3_conditioning_v2'))
    ap.add_argument('--worker');ap.add_argument('--backend',choices=['native','cantera'])
    args=ap.parse_args();root=args.input.resolve();out=args.output.resolve()
    m=runner.read(root/'manifest.json')
    if m['environment']!=runner.environment():raise ValueError('Campaign code/environment differs')
    selected=[c for c in m['cases'] if c['phi']==1 and c['pressure_atm']==1 and c['temperature']==300
              and ((c['transport']=='mixture-averaged' and not c['soret']) or
                   (c['transport']=='multicomponent' and c['soret']))]
    if args.worker:
        worker(root,out,next(c for c in selected if c['id']==args.worker),args.backend);return
    out.mkdir(parents=True,exist_ok=True)
    provenance=dict(parent_sha256=runner.digest(root/'manifest.json'),script_sha256=runner.digest(Path(__file__)),cases=selected)
    if (out/'manifest.json').exists() and runner.read(out/'manifest.json')!=provenance:raise ValueError('Incompatible conditioning directory')
    runner.atomic_json(out/'manifest.json',provenance)
    for c in selected:
        for backend in ('native','cantera'):
            folder=out/c['id']/backend;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'conditioning.json').exists():continue
            print('Conditioning:',c['id'],backend,flush=True)
            with (folder/'console.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--input',str(root),'--output',str(out),
                                '--worker',c['id'],'--backend',backend],cwd=ROOT,env=dict(os.environ,**runner.THREADS),
                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)


if __name__=='__main__':main()
