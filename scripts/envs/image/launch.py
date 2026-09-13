"""Launch one bounded build or validation step in an existing allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--job-id', required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--script', type=Path, required=True)
parser.add_argument('--environment', type=Path)
parser.add_argument('--cpus', type=int, default=16)
parser.add_argument('--minutes', type=int, default=45)
parser.add_argument('--gpus', type=int)
args = parser.parse_args()
assert args.name.replace('-', '').replace('_', '').isalnum()
out = args.out.resolve()
assert out.is_dir()
prefixes = ('SLURM_EDF_', 'OCI_ANNOTATION_', 'PYXIS_', 'SLURM_CPU_BIND', 'ENROOT_',
            'SRUN_CONTAINER', 'SRUN_ENVIRONMENT', 'SBATCH_CONTAINER', 'SBATCH_ENVIRONMENT', 'SLURM_CONTAINER')
names = {'LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV',
         'HF_HUB_ENABLE_HF_TRANSFER', 'PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK', 'SLURM_EXPORT_ENV'}
def excluded(key):
    return 'SPANK' in key or key.startswith(prefixes) or key in names
env = {k: v for k, v in os.environ.items() if not excluded(k)}
command = ['srun', '--jobid=' + args.job_id, '--overlap', '--cpu-bind=none', '--nodes=1', '--ntasks=1',
           '--cpus-per-task=' + str(args.cpus), '--time=' + str(args.minutes)]
if args.environment:
    command.append('--environment=' + str(args.environment.resolve()))
if args.gpus is not None:
    command.append('--gpus-per-task=' + str(args.gpus))
command += ['bash', str(args.script.resolve()), str(out)]
plan = {'command': command, 'removed_environment_keys': sorted(k for k in os.environ if excluded(k))}
(out / (args.name + '-launch.json')).write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(plan, indent=2), flush=True)
with (out / (args.name + '.log')).open('x') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
(out / (args.name + '-exit.txt')).write_text(str(result.returncode) + '\n')
print('Exit:', result.returncode, 'Output:', out, flush=True)
print('\n'.join((out / (args.name + '.log')).read_text().splitlines()[-25:]), flush=True)
raise SystemExit(result.returncode)
