import subprocess
import time
import os

from helpers import utils

print(os.environ['CUDA_VISIBLE_DEVICES'])

command = 'srun --ntasks=1 --gpus=1 --gpu-bind=per_task:{gpus} --cpus-per-task=2 --mem=20G python3 test_srun.py'

t0 = time.time()
processes = []
for i in range(5):
    # subprocess.run(command.format(gpus=1).split(' '))
    # p = subprocess.Popen([os.path.join(utils.ROOT_FOLDER, 'wrapper.sh'), f'{i}'])
    p = subprocess.Popen(command.format(gpus=1).split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes.append(p)

for p in processes:
    p.wait()

dt = time.time() - t0
print(f'Everything done in {dt:.2f} s')