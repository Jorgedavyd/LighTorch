# https://github.com/pytorch/pytorch/blob/main/torch/utils/bottleneck/__main__.py
# adapted for decorators framework
from timeit import default_timer as timer
from torch import nn, Tensor
from typing import Sequence, Any, Callable, Dict
from torch.utils.benchmark import 
from torch.utils.bottleneck.__main__ import run_env_analysis
import cProfile
import pstats
import sys
import os

import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info

def compiled_with_cuda(sysinfo):
    if sysinfo.cuda_compiled_version:
        return f'compiled w/ CUDA {sysinfo.cuda_compiled_version}'
    return 'not compiled w/ CUDA'

env_summary = """
--------------------------------------------------------------------------------
  Environment Summary
--------------------------------------------------------------------------------
PyTorch {pytorch_version}{debug_str} {cuda_compiled}
Running with Python {py_version} and {cuda_runtime}

`{pip_version} list` truncated output:
{pip_list_output}
""".strip()

def run_env_analysis():
    print('Running environment analysis...')
    info = get_env_info()

    result: Dict[str, str] = {}

    debug_str = ''
    if info.is_debug_build:
        debug_str = ' DEBUG'

    cuda_avail = ''
    if info.is_cuda_available:
        cuda = info.cuda_runtime_version
        if cuda is not None:
            cuda_avail = 'CUDA ' + cuda
    else:
        cuda = 'CUDA unavailable'

    pip_version = info.pip_version
    pip_list_output = info.pip_packages
    if pip_list_output is None:
        pip_list_output = 'Unable to fetch'

    result = {
        'debug_str': debug_str,
        'pytorch_version': info.torch_version,
        'cuda_compiled': compiled_with_cuda(info),
        'py_version': f'{sys.version_info[0]}.{sys.version_info[1]}',
        'cuda_runtime': cuda_avail,
        'pip_version': pip_version,
        'pip_list_output': pip_list_output,
    }

    return env_summary.format(**result)

def run_cprofile(code, globs, launch_blocking=False):
    print('Running your script with cProfile')
    prof = cProfile.Profile()
    prof.enable()
    exec(code, globs, None)
    prof.disable()
    return prof

cprof_summary = """
--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
""".strip()

def print_cprofile_summary(prof, sortby='tottime', topk=15):
    print(cprof_summary)
    cprofile_stats = pstats.Stats(prof).sort_stats(sortby)
    cprofile_stats.print_stats(topk)
    
def run_autograd_prof(code, globs):
    def run_prof(use_cuda=False):
        with profiler.profile(use_cuda=use_cuda) as prof:
            exec(code, globs, None)
        return prof

    print('Running your script with the autograd profiler...')
    result = [run_prof(use_cuda=False)]
    if torch.cuda.is_available():
        result.append(run_prof(use_cuda=True))
    else:
        result.append(None)

    return result

# Set of tools for module testing and computational analysis

# Bottleneck decorator for forward pass, defined from torch.utils.bottleneck
def bottleneck(func: Callable) -> None:
    pass

# timer decorator to assess computational time of runtime
def timer(func: Callable) -> None:
    pass



def compare(module1: nn.Module, module2: nn.Module) -> None:
    

__all__ = ['bottleneck', 'timer']