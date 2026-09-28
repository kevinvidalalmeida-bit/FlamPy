"""Spatial study checks use synthetic rows; do not resolve any flames."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import postprocess_flame_mesh_sensitivity as post
spec=importlib.util.spec_from_file_location('spatial_runner',ROOT/'benchmarks/benchmark_flame_mesh_sensitivity.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def test_six_states_match_campaign_transport_and_count():
    cases=study.anchors()
    assert len(cases)==len({c['id'] for c in cases})==6
    assert all(c['phi']==1 for c in cases)
    assert all(c['soret']==(c['fuel']=='H2') for c in cases)
    assert len(cases)*(3+1)*2==48


def test_signed_changes_missing_failures_and_domain():
    c=dict(id='synthetic',fuel='H2',pressure_atm=1,temperature=300)
    base=[dict(case_id='synthetic',backend=b,Su=2.,usable=True,nodes=100,width=.06,status='accepted')
          for b in ('native','cantera')]
    m=dict(cases=[c],baseline=base,levels=[2,3,4,5])
    records=[dict(base[0],phase='verify-L4',settings=dict(level=4),Su=1.98),
             dict(base[0],phase='verify-L5',settings=dict(level=5),Su=1.99),
             dict(base[0],phase='domain-L3',settings=dict(level=3),Su=2.002),
             dict(base[1],phase='verify-L4',settings=dict(level=4),usable=False,status='failed')]
    rows=post.analyze(m,records)
    assert len(rows)==10
    lookup={(r['backend'],r['test']):r for r in rows}
    assert lookup['native','L4']['signed_change_L3_percent']<0
    assert lookup['native','L5']['successive_change_percent']>0
    assert lookup['cantera','L4']['status']=='failed'
    assert lookup['cantera','L4']['signed_change_L3_percent'] is None
    assert lookup['native','L2']['status']=='pending'
    assert lookup['native','domain-L3']['signed_change_L3_percent']>0


def test_partial_report_exports_only_available_data(tmp_path,monkeypatch):
    import json
    cases=study.anchors()
    base=[dict(case_id=c['id'],backend=b,Su=2.,usable=True,nodes=100,width=.06,status='accepted')
          for c in cases for b in ('native','cantera')]
    root=tmp_path/'synthetic';root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(kind='L3_spatial_sensitivity',cases=cases,baseline=base,levels=[2,3,4,5])))
    monkeypatch.setattr(post.pp,'load_records',lambda root:[
        dict(base[0],phase='verify-L4',settings=dict(level=4),Su=1.99)])
    out=tmp_path/'synthetic_report'
    post.main(['--input',str(root),'--output',str(out)])
    report=json.loads((out/'sensibilidad_L3.json').read_text())
    assert report['usable']==13 and report['new_usable']==1
    assert (out/'11_sensibilidad_L3.pdf').is_file()
    assert (out/'11_sensibilidad_L3.png').is_file()
    assert '13/60' not in (out/'sensitivity_report.tex').read_text()  # Caption stays self-contained.
    assert sum(r['status']=='pending' for r in report['rows'])==47
