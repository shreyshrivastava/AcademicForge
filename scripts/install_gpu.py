"""Install application dependencies without replacing a VM's CUDA/ROCm stack."""

import argparse
from pathlib import Path
import subprocess
import sys


def verify(kind):
    try:
        import torch
        from packaging.version import Version
    except ImportError as exc:
        raise RuntimeError('Install the GPU image\'s supported PyTorch wheel first.') from exc
    if not (Version('2.12') <= Version(torch.__version__.split('+')[0]) < Version('2.13')):
        raise RuntimeError('This dependency profile targets PyTorch 2.12.x. Use a matching verified GPU image.')
    accelerator = 'rocm' if torch.version.hip else 'cuda'
    if not torch.cuda.is_available() or accelerator != kind:
        raise RuntimeError(f'The installed PyTorch stack does not expose the requested {kind} GPU.')
    print(f'Using existing {torch.__version__} on {torch.cuda.get_device_name(0)} ({kind})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['cuda', 'rocm'])
    args = parser.parse_args()
    try:
        verify(args.kind)
    except RuntimeError as exc:
        parser.exit(1, str(exc) + '\n')
    requirements = Path(__file__).resolve().parents[1] / 'requirements-gpu.txt'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-r', str(requirements)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    verify(args.kind)


if __name__ == '__main__':
    main()
