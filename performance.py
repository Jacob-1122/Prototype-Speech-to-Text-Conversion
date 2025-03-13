import os
import time
import psutil
import torch
import logging

logger = logging.getLogger(__name__)

def log_performance_usage(step_name, start_time):
    """Logs elapsed time, memory usage, and CPU/GPU usage for a given step."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    cpu_usage = psutil.cpu_percent(interval=1)
    elapsed = time.time() - start_time
    logger.info(f"[{step_name}] Elapsed Time: {elapsed:.2f}s | Memory Usage: {mem_usage:.2f} MB | CPU Usage: {cpu_usage:.2f}%")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.info(f"[{step_name}] GPU Memory Allocated: {gpu_mem:.2f} MB")
