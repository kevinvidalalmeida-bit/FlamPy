"""The thread policy holds whatever the import order; explicit settings win."""
import json
import os
import subprocess
import sys

import pytest

SCRIPT = (
    "import json, numpy, numba\n"
    "import kflame\n"
    "print(json.dumps(kflame.effective_threads()))\n"
)


def _run(env_update):
    env = {k: v for k, v in os.environ.items() if k not in ('OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')}
    env.update(env_update)
    out = subprocess.run([sys.executable, '-c', SCRIPT], capture_output=True, text=True, env=env, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_numpy_and_numba_imported_first_still_get_the_policy():
    pytest.importorskip('threadpoolctl')
    threads = _run({})
    assert threads['numba'] == 4
    assert all(count == 1 for count in threads['blas'].values())


def test_explicit_environment_overrides_are_respected():
    pytest.importorskip('threadpoolctl')
    threads = _run({'OPENBLAS_NUM_THREADS': '2', 'NUMBA_NUM_THREADS': '3'})
    assert threads['numba'] == 3
    assert all(count == 2 for count in threads['blas'].values())
