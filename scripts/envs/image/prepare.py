"""Stage checked offline image inputs from this checkout's dependency pins."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(packages: Path, out: Path):
    recipe = Path(__file__).resolve().parent
    repo = recipe.parents[2]
    manifest = json.loads((recipe / 'packages.lock.json').read_text())
    relative = 'src/3rdparty/lhotse'
    source = repo / relative
    entry = git(repo, 'ls-files', '--stage', '--', relative).split()
    expected = manifest['source_revisions']['lhotse']
    if len(entry) != 4 or entry[:3] != ['160000', expected, '0']:
        raise ValueError('Lhotse gitlink and packages.lock.json must pin the same commit')
    if not (source / '.git').exists():
        raise ValueError('Run git submodule update --init src/3rdparty/lhotse first')
    if git(source, 'rev-parse', 'HEAD') != expected or git(source, 'status', '--porcelain'):
        raise ValueError('Lhotse checkout must be clean and match the recorded gitlink')
    if out.exists():
        raise FileExistsError(out)
    manifest['source_revisions']['pipeline'] = git(repo, 'rev-parse', 'HEAD')
    manifest['pipeline_dirty'] = bool(git(repo, 'status', '--porcelain'))
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish the manifest and packages together, only after every hash passes.
    with tempfile.TemporaryDirectory(prefix=out.name + '.', dir=out.parent) as temporary:
        staged = Path(temporary) / 'build'
        destination = staged / 'packages'
        destination.mkdir(parents=True)
        sums = []
        for package in manifest['packages']:
            name = package['file']
            if Path(name).name != name:
                raise ValueError(f'Package filename must be a basename: {name}')
            target = destination / name
            shutil.copyfile(packages / name, target)
            if target.stat().st_size != package['bytes'] or sha256(target) != package['sha256']:
                raise ValueError(f'Package size or SHA256 mismatch: {name}')
            sums.append(f"{package['sha256']}  {name}\n")
        (destination / 'SHA256SUMS').write_text(''.join(sums))
        shutil.copytree(recipe, staged / 'recipe', ignore=shutil.ignore_patterns('__pycache__'))
        (staged / 'package-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staged.rename(out)
    print(f'Staged {len(sums)} verified inputs; Lhotse {expected}; output {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.packages.resolve(), args.out.resolve())
