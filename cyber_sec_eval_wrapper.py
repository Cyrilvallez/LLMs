import argparse
import time

import torch

from TextWiz import textwiz
from helpers import utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CyberSecEvalInstruct benchmark')
    parser.add_argument('--dataset', type=str, choices=['original', 'llama3', 'llama3_python_only'], default='llama3_python_only',
                        help='Version of the CyberSecEval instruct benchmark (i.e. model used for reformulations)')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    
    args = parser.parse_args()
    dataset = args.dataset
    int8 = args.int8
    int4 = args.int4

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')

    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")
    
    num_gpus = torch.cuda.device_count()

    # Select models
    models = [
        'zephyr-7B-beta',
        'mistral-7B-instruct-v2',
        'starling-7B-beta',
        'star-chat-alpha',
        'llama3-8B-instruct',
        'command-r',
        'code-llama-34B-instruct',
        'llama2-70B-chat',
        'code-llama-70B-instruct',
        'llama3-70B-instruct',
    ]

    print(f'Launching computations with {num_gpus} gpus available.')

    gpu_footprints = textwiz.estimate_number_of_gpus(models, int8, int4)
    # Skip models needing more resources than we have
    skipped_models = [model for model, gpu in zip(models, gpu_footprints) if gpu > num_gpus]
    print(f'Skipping {*skipped_models,} due to lack of gpu resources.')

    # Only pick those that needs at most the resources we have
    models = [model for model, gpu in zip(models, gpu_footprints) if gpu <= num_gpus]
    gpu_footprints = [gpu for gpu in gpu_footprints if gpu <= num_gpus]

    # Create the commands to run
    commands = [f'python3 -u cyber_sec_eval.py {model} --dataset {dataset}' for model in models]
    if int8:
        commands = [c + ' --int8' for c in commands]
    if int4:
        commands = [c + ' --int4' for c in commands]
        
    t0 = time.time()

    utils.dispatch_jobs_srun(gpu_footprints, num_gpus, commands)

    dt = time.time() - t0
    print(f'Overall it took {dt/3600:.2f}h !')

