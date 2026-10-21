"""Regenerate pip exports from the committed universal uv lockfile."""

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    common = ['uv', 'export', '--locked', '--no-emit-project', '--format', 'requirements-txt']
    subprocess.run(common + ['--extra', 'local', '--no-dev', '-o', 'requirements-local.txt'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(common + ['-o', 'requirements-dev.txt'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    lock = tomllib.loads((ROOT / 'uv.lock').read_text())
    excluded = [p['name'] for p in lock['package'] if p['name'] in {'torch', 'triton'} or p['name'].startswith(('nvidia-', 'cuda-', 'mlx'))]
    args = common + ['--no-dev', '-o', 'requirements-gpu.txt']
    for package in excluded:
        args.extend(['--no-emit-package', package])
    subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
