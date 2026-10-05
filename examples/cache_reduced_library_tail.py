"""Reuse identical reactor endpoints and compute new tails in processes.

Reuse requires identical physical endpoint, mechanism, sampling and progress
weights. The source-table digest binds the resulting cache to its new library.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM
from kflame.chemistry.mechanism import load_mechanism
from kflame.chemistry.kinetics import NativeKinetics
from kflame.chemistry.initialization import hp_equilibrium
from examples.extend_reduced_fgm_tail import reactor_tail


def initialize(source):
    global model,mech,kinetics
    model=NonAdiabaticFGM(source);mech=load_mechanism(model.metadata['mechanism']);kinetics=NativeKinetics(mech)


def calculate(index):
    index=int(index)
    shape=tuple(model.table['structured_shape']);n=shape[2]-1+index*shape[2]
    Y=model.table['Y'][n];T=model.table['T'][n];h=model.table['h'][n]
    seed=np.maximum(Y,0.);seed/=seed.sum()
    _,equilibrium,_=hp_equilibrium(mech,float(T),model.metadata['pressure_Pa'],seed)
    try:
        tail,audit=reactor_tail(model,mech,kinetics,T,Y,h,equilibrium,16,endpoint_policy='monotone_branch')
    except (RuntimeError,ValueError) as error:
        return index,None,dict(error=str(error),index=index,temperature_K=float(T),
            progress_gain=float(model.table['progress_weights']@(equilibrium-Y)),
            species_gap=float(np.max(abs(equilibrium-Y))))
    audit['HP_Y']=equilibrium.tolist()
    return index,tail,audit


def run(source,previous,output,jobs=4):
    current=NonAdiabaticFGM(source);old=NonAdiabaticFGM(previous)
    if (current.metadata['mechanism_sha256']!=old.metadata['mechanism_sha256']
            or not np.array_equal(current.table['progress_weights'],old.table['progress_weights'])):
        raise ValueError('Mechanism and progress must match for reactor reuse')
    shape=tuple(current.table['structured_shape']);old_shape=tuple(old.table['structured_shape'])
    count=shape[0]*shape[1];tails=np.empty((count,16,len(current.table['species_names'])));audits=[None]*count
    old_audits=json.loads((Path(previous)/'reactor_audit.json').read_text(encoding='utf-8'))
    hashes={old.metadata['training_profile_sha256'][r['output']]:r['composition_index']*old_shape[1]+r['loss_index'] for r in old.metadata['rows']}
    pending=[];reused=0;HP_Y=np.empty((count,len(current.table['species_names'])))
    with np.load(Path(previous)/'equilibrium_endpoints.npz',allow_pickle=False) as saved:
        old_HP=saved['Y']
    for row in current.metadata['rows']:
        index=row['composition_index']*shape[1]+row['loss_index'];digest=current.metadata['training_profile_sha256'][row['output']]
        candidate=hashes.get(digest);n=(index+1)*shape[2]-1
        if candidate is not None:
            old_n=candidate*old_shape[2]+shape[2]-1
            same=np.array_equal(current.table['Y'][n],old.table['Y'][old_n]) and abs(current.table['h'][n]-old.table['h'][old_n])<1e-8
            if same and old_audits[candidate].get('endpoint_policy')=='monotone_branch':
                tails[index]=old.table['Y'][candidate*old_shape[2]+shape[2]:(candidate+1)*old_shape[2]]
                audits[index]=old_audits[candidate];HP_Y[index]=old_HP[candidate];reused+=1;continue
        pending.append(index)
    failures=[]
    with ProcessPoolExecutor(max_workers=jobs,initializer=initialize,initargs=(source,)) as pool:
        for position,(index,tail,audit) in enumerate(pool.map(calculate,pending)):
            if tail is None:
                failures.append(audit);print('TAIL_FAILURE',audit,flush=True);continue
            tails[index],audits[index]=tail,audit
            HP_Y[index]=audit.pop('HP_Y')
            if position%10==0:print('TAIL',position,len(pending),flush=True)
    if failures:
        Path(output).with_suffix('.failures.json').write_text(json.dumps(failures,indent=2)+'\n',encoding='utf-8')
        raise RuntimeError('Reactor cache rejected: unresolved chemical tails; see failure diagnostics')
    np.savez_compressed(output,Y=tails,HP_Y=HP_Y,source_table_sha256=current.table_sha256,audit_json=json.dumps(audits),reused_tails=reused)
    print(dict(reused=reused,new=len(pending)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for q in ['source','previous','output']:p.add_argument('--'+q,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4);a=p.parse_args();run(a.source,a.previous,a.output,a.jobs)
