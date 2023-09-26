import multiprocessing as mp
import time
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import torch
from transformers import GenerationConfig

import engine
from engine import stopping
from engine import loader
from engine.prompt_template import PROMPT_MODES
from engine.code_parser import CodeParser, PythonParser
from helpers import datasets
from helpers import utils


@utils.duplicate_function_for_gpu_dispatch
def target(name: str, foo, bar = 3):
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(f'Number of gpus seen by torch: {torch.cuda.device_count()}')
    # model = engine.HFModel(name)
    # print(f'Gpus as seen by torch: {model.get_gpu_devices()}')


@utils.duplicate_function_for_gpu_dispatch
def sleep(dt: float = 4):
    time.sleep(dt)
    print(f'Number of gpus seen by torch: {torch.cuda.device_count()}')


if __name__ == '__main__':

    # num_gpus = torch.cuda.device_count()
    num_gpus = 3
    gpu_footprints = [1,1,1,2,2,1,3]
    # dt = [4]*len(gpu_footprints)


    t0 = time.time()
    utils.dispatch_jobs_pool(gpu_footprints, num_gpus, 0.01, sleep, dt)
    # utils.dispatch_jobs_pool(gpu_footprints, num_gpus, 0.01, sleep)
    dt0 = time.time() - t0

    print(f'Time with a pool: {dt0:.2f} s')

    t1 = time.time()
    utils.dispatch_jobs(gpu_footprints, num_gpus, 0.01, sleep, dt)
    # utils.dispatch_jobs(gpu_footprints, num_gpus, 0.01, sleep)
    dt1 = time.time() - t1

    print(f'Time without a pool: {dt1:.2f} s')
    