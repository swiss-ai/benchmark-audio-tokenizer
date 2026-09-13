import importlib.metadata as metadata
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

out = Path('/build')
before = json.loads((out / 'python-before.json').read_text())
after = json.loads((out / 'python-after.json').read_text())
for key in ('python', 'torch', 'torch_file', 'cuda', 'cudnn'):
    assert before[key] == after[key], (key, before[key], after[key])
assert after['torch'] == '2.13.0a0+8145d630e8.nv26.06'
assert after['cuda'] == '13.3' and after['cudnn'] == 92401
a = {tuple(x) for x in before['distributions']}
b = {tuple(x) for x in after['distributions']}
assert not a - b, a - b
added = b - a
manifest = json.loads((out / 'package-manifest.json').read_text())
expected = {(canonicalize_name(p['name']), p['version']) for p in manifest['packages'] if p['file'].endswith('.whl')}
actual = {(canonicalize_name(n), v) for n, v, _ in added}
assert len(added) == len(expected) and actual == expected, (actual, expected)
assert (out / 'dpkg-before.tsv').read_bytes() == (out / 'dpkg-after.tsv').read_bytes()
for name, version in expected:
    assert metadata.version(name) == version
    for raw in metadata.requires(name) or []:
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({'extra': ''}):
            continue
        assert metadata.version(req.name) in req.specifier, (name, raw, metadata.version(req.name))
import lhotse
import torchaudio
import torch
assert Path(lhotse.__file__).is_relative_to('/opt/venv/lib/python3.12/site-packages')
assert Path(torchaudio.__file__).is_relative_to('/opt/venv/lib/python3.12/site-packages')
assert torch.ops._torchaudio.cuda_version() == 13030
(out / 'package-diff.json').write_text(json.dumps({'added': sorted(added), 'removed': [], 'system_packages_changed': [], 'core_preserved': True}, indent=2) + '\n')
print('Verified offline additions, dependency closure, CUDA version and unchanged core runtime.')
