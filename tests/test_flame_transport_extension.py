"""Synthetic protocol tests; never launch scientific flame solves."""
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
sys.path.insert(0, str(ROOT/'tools'))
import extend_flame_transport_sweeps as extension
import postprocess_flame_sweeps as post


def test_full_design_has_104_unique_conditions_and_48_new():
    base = extension.runner.cases()
    extra = extension.additional_cases()
    assert len(extra) == 48
    assert len({r['id'] for r in base+extra}) == 104
    assert not ({r['id'] for r in base} & {r['id'] for r in extra})
    for fuel in ('CH4', 'H2'):
        assert sum(c['fuel'] == fuel for c in extra) == 24
        for mode in extension.runner.MODES:
            rr = [c for c in base+extra if c['fuel'] == fuel and (c['transport'], c['soret']) == mode]
            assert {c['pressure_atm'] for c in rr if c['phi'] == 1 and c['temperature'] == 300} == {1, 2, 3, 5, 10}
            assert {c['temperature'] for c in rr if c['phi'] == 1 and c['pressure_atm'] == 1} == {300, 350, 400, 450, 500}
            assert len(rr) == 13
            assert all(c['native_strategy'] == ('ptc-ser' if fuel == 'CH4' else 'backward-euler') for c in rr)


def seed(tmp_path, monkeypatch):
    runner = extension.runner
    base = dict(cases=runner.cases(), pairs=5, environment={'synthetic': True}, bootstrap_seed=20260927,
                smoke=False, spatial_protocol={'selected_level':3})
    runner.atomic_json(tmp_path/'manifest.json', base)
    settings = runner.settings(3)
    monkeypatch.setattr(extension, 'preflight', lambda root: (base, settings))
    return base, settings


def test_dry_run_is_read_only_and_rejects_incompatible_extension(tmp_path, monkeypatch):
    base, settings = seed(tmp_path, monkeypatch)
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert extension.main(['--campaign',str(tmp_path),'--dry-run']) == 0
    assert before == {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    ext = extension.extension_manifest(tmp_path, base, settings)
    ext['settings'] = dict(settings, slope=.2)
    extension.runner.atomic_json(tmp_path/'transport_extension.json', ext)
    with pytest.raises(ValueError, match='Incompatible transport extension'):
        extension.main(['--campaign',str(tmp_path),'--resume'])


def test_offline_manifest_combines_without_changing_base(tmp_path, monkeypatch):
    base, settings = seed(tmp_path, monkeypatch)
    original = (tmp_path/'manifest.json').read_bytes()
    ext = extension.extension_manifest(tmp_path, base, settings)
    extension.runner.atomic_json(tmp_path/'transport_extension.json', ext)
    combined = post.load_manifest(tmp_path)
    assert len(combined['cases']) == 104
    assert (tmp_path/'manifest.json').read_bytes() == original
    ext['cases'][0] = base['cases'][0]
    extension.runner.atomic_json(tmp_path/'transport_extension.json', ext)
    with pytest.raises(ValueError, match='duplicate/shared'):
        post.load_manifest(tmp_path)
    ext['base_manifest_sha256'] = 'wrong'
    extension.runner.atomic_json(tmp_path/'transport_extension.json', ext)
    with pytest.raises(ValueError, match='different base'):
        post.load_manifest(tmp_path)


def test_extension_preserves_failures_and_restarts_interrupted_pairs(tmp_path, monkeypatch):
    seed(tmp_path, monkeypatch)
    one = extension.additional_cases()[:1]
    monkeypatch.setattr(extension, 'additional_cases', lambda: one)
    runner = extension.runner
    calls = []
    class Process:
        def __init__(self, argv, **kwargs):
            self.folder = Path(argv[-1]).parent
            self.number = len(calls)
            calls.append(runner.read(argv[-1]))
            runner.atomic_json(self.folder/'result.json', dict(status='rejected',
                accepted=False, diagnostics_complete=True, time_s=1.))
        def wait(self, timeout=None):
            if timeout and self.number == 1:
                raise KeyboardInterrupt()
            return 0
        def kill(self):
            pass
    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    argv = ['--campaign', str(tmp_path), '--resume']
    with pytest.raises(KeyboardInterrupt):
        extension.main(argv)
    assert len(calls) == 2
    assert extension.main(argv) == 2  # Completed rejected pairs are retained.
    assert len(calls) == 12  # Interrupted pair restarted in full, then four pairs.
    assert extension.main(argv) == 2
    assert len(calls) == 12
    assert len(list(tmp_path.glob('main/*/pair-00/attempt-*'))) == 2
    assert all(c['case'] == one[0] and c['settings']['level'] == 3 for c in calls)
    assert {c['backend'] for c in calls} == {'native','cantera'}


def test_extended_physical_and_speedup_have_all_four_curves(monkeypatch, tmp_path):
    import redesign_thesis_result_figures as figures
    import matplotlib.pyplot as plt
    rows = [dict(c, ratio=2., ci=[1.8, 2.2], Su_native=1., Su_cantera=1.)
            for c in extension.runner.cases()+extension.additional_cases()]
    saved = {}
    monkeypatch.setattr(figures, 'save', lambda fig, out, name, caption: saved.update({name: (fig, caption)}))
    figures.physical_figure(rows, tmp_path)
    figures.speedup_figure(rows, tmp_path)
    physical, caption = saved['13_respuesta_fisica']
    assert all(len(ax.lines) == 8 and all(len(line.get_xdata()) == 5 for line in ax.lines) for ax in physical.axes)
    speedup, caption = saved['14_speedup_global']
    # Four continuous transport curves, plus markers and the equality reference.
    assert all(sum(len(line.get_xdata()) == 5 for line in ax.lines) == 4 for ax in speedup.axes)
    assert 'tres barridos incluyen las cuatro' in caption
    plt.close('all')


def test_new_postprocessor_does_not_import_solver():
    import ast
    tree = ast.parse((ROOT/'tools/postprocess_extended_flames.py').read_text(encoding='utf-8'))
    imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(s.startswith(('kflame', 'cantera', 'subprocess', 'extend_flame')) for s in imports)


def test_partial_offline_report_keeps_pending_conditions(tmp_path, monkeypatch):
    import postprocess_extended_flames as extended_post
    base, settings = seed(tmp_path, monkeypatch)
    extension.runner.atomic_json(tmp_path/'transport_extension.json',
                                extension.extension_manifest(tmp_path, base, settings))
    assert extended_post.main(['--input',str(tmp_path),'--output',str(tmp_path/'report_expanded')]) == 0
    report = post.read(tmp_path/'report_expanded/summary.json')
    assert report['expected_main'] == 1040
    assert report['recorded_main'] == report['usable_main'] == 0
    assert len(report['pending']) == 104
    assert all(p['missing_pairs'] == 5 for p in report['pending'])
    assert not (tmp_path/'report_expanded/revision_figures').exists()


def test_full_timing_details_preserve_five_observations(monkeypatch, tmp_path):
    import redesign_thesis_result_figures as figures
    import matplotlib.pyplot as plt
    stats = dict(times=[1.,1.1,1.2,1.3,1.4], median_s=1.2, quartiles_s=[1.1,1.3])
    rows = [dict(c, timing={b:stats for b in figures.SOLVERS})
            for c in extension.runner.cases()+extension.additional_cases()]
    saved = {}
    monkeypatch.setattr(figures,'save',lambda fig,out,name,caption:saved.update({name:fig}))
    figures.timing_figures(rows, tmp_path)
    assert set(saved) == {'09_tiempos_transportes','09_tiempos_presion','09_tiempos_temperatura'}
    for name in ('09_tiempos_presion','09_tiempos_temperatura'):
        assert len(saved[name].axes) == 8
        assert all(len(ax.lines) == 2 and all(len(line.get_xdata()) == 5 for line in ax.lines)
                   for ax in saved[name].axes)
    plt.close('all')
