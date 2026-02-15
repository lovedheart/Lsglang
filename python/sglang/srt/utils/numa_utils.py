import logging
import multiprocessing
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@contextmanager
def configure_subprocess(server_args: ServerArgs, gpu_id: int, tp_size: int):
    from sglang.srt.utils.common import is_numa_interleave_enabled
    import subprocess
    import re
    if (
        numa_nodes := server_args.numa_node
    ) is not None and envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = numa_nodes[gpu_id]
        numactl_args = f"--cpunodebind={numa_node} --membind={numa_node}"
        executable, debug_str = _create_numactl_executable(numactl_args=numactl_args)
        with _mp_set_executable(executable=executable, debug_str=debug_str):
            yield
    elif is_numa_interleave_enabled():
        try:
            result = subprocess.run(['numactl', '--hardware'], capture_output=True, text=True)
            match = re.search(r'available: (\d+) nodes', result.stdout)
            num_numa_nodes = int(match.group(1)) if match else 1
        except:
            num_numa_nodes = 1
         
        num_gpus = tp_size
        
        if num_numa_nodes >= num_gpus: 
            nodes_per_gpu = num_numa_nodes // num_gpus
            start_node = gpu_id * nodes_per_gpu
            if gpu_id == num_gpus - 1:
                end_node = num_numa_nodes - 1
            else:
                end_node = start_node + nodes_per_gpu - 1
            
            if start_node == end_node:
                numa_config = str(start_node)
            else:
                numa_config = f"{start_node}-{end_node}"
            mode = "Exclusive Nodes"
        else: 
            gpus_per_node = num_gpus // num_numa_nodes
            node_id = gpu_id // gpus_per_node
            node_id = min(node_id, num_numa_nodes - 1)
            numa_config = str(node_id)
            mode = "Shared Nodes"
        
        logger.info(f"gpu_id: {gpu_id}, tp_size: {tp_size}, numa_config: {numa_config}, mode: {mode}")
        numactl_args = f"--cpunodebind={numa_config} --membind={numa_config}"
        executable, debug_str = _create_numactl_executable(numactl_args=numactl_args)
        with _mp_set_executable(executable=executable, debug_str=debug_str):
            yield
    else:
        yield


def _create_numactl_executable(numactl_args: str):
    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())
    script = f'''#!/bin/sh
exec numactl {numactl_args} {old_executable} "$@"'''
    path = Path(
        f"/tmp/sglang_temp_file_{time.time()}_{random.randrange(0, 10000000)}.sh"
    )
    path.write_text(script)
    path.chmod(0o777)
    return str(path), f"{script=}"


@contextmanager
def _mp_set_executable(executable: str, debug_str: str):
    start_method = multiprocessing.get_start_method()
    assert start_method == "spawn", f"{start_method=}"

    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())
    multiprocessing.spawn.set_executable(executable)
    logger.info(f"mp.set_executable {old_executable} -> {executable} ({debug_str})")
    try:
        yield
    finally:
        assert (
            os.fsdecode(multiprocessing.spawn.get_executable()) == executable
        ), f"{multiprocessing.spawn.get_executable()=}"
        multiprocessing.spawn.set_executable(old_executable)
        logger.info(f"mp.set_executable revert to {old_executable}")
