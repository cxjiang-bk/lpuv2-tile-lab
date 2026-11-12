#!/usr/bin/env python3
"""Apply the configured build-job cap to native runtime child builds."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

real = os.environ['LPUV2_REAL_CMAKE']
args = sys.argv[1:]
if args and args[0] == '--build':
    capped = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ('--parallel', '-j'):
            i += 1
            if i < len(args) and args[i].isdigit():
                i += 1
        elif arg.startswith('--parallel=') or (arg.startswith('-j') and arg[2:].isdigit()):
            i += 1
        else:
            capped.append(arg)
            i += 1
    insertion = capped.index('--') if '--' in capped else len(capped)
    capped[insertion:insertion] = ['--parallel', os.environ.get('PYPTO_BUILD_JOBS', '2')]
    args = capped
    # Simpler may launch several cache builds concurrently. Share one lock.
    if '/runtime/build/cache/' in str(Path.cwd()) + '/':
        with open(os.environ['LPUV2_CMAKE_LOCK'], 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            sys.exit(subprocess.call([real, *args]))
os.execv(real, [real, *args])
