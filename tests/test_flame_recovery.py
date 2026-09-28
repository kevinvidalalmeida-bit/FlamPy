"""Offline classification tests; synthetic trajectories do not enter thesis data."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from postprocess_flame_recovery import classify
import postprocess_flame_recovery as recovery


def record(histories, *, energy=True):
    return dict(instrumentation=dict(phases=[dict(energy=energy, history=histories)]),
                report=dict(refine=[]), width=.03, settings=dict(width=.03))


def test_configured_ptc_does_not_mean_used():
    result = record([dict(phase='steady', ok=True)])
    result['effective_options'] = dict(pseudo_time_method='ptc-ser')
    flags, counts = classify(result)
    assert flags['newton_only'] and not flags['ptc'] and not flags['be']
    assert counts['steady_attempts'] == 1


def test_rejected_attempts_and_interrupted_bootstrap_are_counted_once():
    result = record([dict(phase='steady', ok=True)])
    result['instrumentation']['phases'].insert(0, dict(energy=True,
        interrupted='DomainTooNarrowError', history=[
            dict(phase='transient', scheme='PTC-SER', ok=False),
            dict(phase='transient', scheme='BE-fallback', ok=True)]))
    result['report']['transport_bootstrap'] = dict(refine=[dict(changed=True)])
    result['report']['solver_trace'] = result['instrumentation']['phases']
    result['width'] = .06
    flags, counts = classify(result)
    assert flags == dict(newton_only=False, ptc=True, be=True, thermal=False, domain=True, mesh=True)
    assert counts['ptc_attempts'] == 1 and counts['ptc_rejected'] == 1
    assert counts['be_accepted'] == 1 and counts['interrupted_domain_phases'] == 1


def test_thermal_recovery_is_not_direct_newton():
    flags, _ = classify(record([dict(phase='steady', ok=True)], energy=False))
    assert flags['thermal'] and not flags['newton_only']
    assert not flags['ptc'] and not flags['be']


def test_missing_history_and_unrecognized_scheme_are_not_silent_zeros():
    with pytest.raises(ValueError, match='Incomplete'):
        classify(record([]))
    with pytest.raises(ValueError, match='Unknown transient'):
        classify(record([dict(phase='transient', scheme='unknown', ok=True)]))


def test_offline_publication_requires_all_104_matching_profiles(tmp_path, monkeypatch):
    """Entirely synthetic fixture in pytest's temporary directory, never campaign output."""
    root, diag, out = [tmp_path / name for name in ('synthetic_production', 'synthetic_diagnostic', 'test_report')]
    root.mkdir(); diag.mkdir()
    (root / 'manifest.json').write_text('{}')
    (root / 'transport_extension.json').write_text('{}')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cases, references = [], {}
    for fuel in ('CH4', 'H2'):
        for mode, (transport, soret) in enumerate(recovery.MODES):
            for i in range(13):
                cid = f'SYNTHETIC_{fuel}_{mode}_{i}'
                cases.append(dict(id=cid, fuel=fuel, transport=transport, soret=soret,
                                  phi=1, pressure_atm=1, temperature=300))
                origin, folder = root / cid, diag / cid / 'native'
                origin.mkdir(); folder.mkdir(parents=True)
                arrays = dict(z=np.array([0., .03]), T=np.array([300., 2000.]),
                              u=np.array([1., 2.]), Y=np.array([[1., 1.]]))
                np.savez(origin / 'profile.npz', **arrays)
                np.savez(folder / 'profile.npz', **arrays)
                references[cid] = dict(folder=cid, profile_sha256=sha(origin / 'profile.npz'))
                value = record([dict(phase='steady', ok=True)])
                value.update(diagnostic_committed=True, usable=True,
                             profile_sha256=sha(folder / 'profile.npz'))
                (folder / 'result.json').write_text(json.dumps(value))
    monkeypatch.setattr(recovery.pp, 'load_manifest', lambda path: dict(cases=cases))
    (diag / 'manifest.json').write_text(json.dumps(dict(kind='flame-recovery-104',
        cases=cases, references=references, parent_manifest_sha256=sha(root / 'manifest.json'),
        extension_sha256=sha(root / 'transport_extension.json'))))
    argv = ['--input', str(root), '--diagnostics', str(diag), '--output', str(out)]
    last = diag / cases[-1]['id'] / 'native/result.json'
    saved = last.read_text(); last.unlink()
    assert recovery.main(argv) == 2
    assert not (out / '17_recuperacion_analisis.tex').exists()
    assert not (out / '17_recuperacion_global.pdf').exists()
    last.write_text(saved)
    assert recovery.main(argv) == 0
    summary = json.loads((out / '17_recuperacion_resumen.json').read_text())
    assert summary['complete'] and summary['available'] == 104
    assert all(g['newton_only'] == 13 and g['ptc'] == 0 for g in summary['groups'])
    assert (out / '17_recuperacion_global.pdf').is_file()
    assert (out / '17_recuperacion_global.png').is_file()
    assert (out / '17_recuperacion_analisis.tex').is_file()
