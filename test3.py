import engine
from engine import stopping
from helpers import datasets
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'facebook/opt-13b'

max_memory = {0: '8GiB', 1: '30GiB'}
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='balanced',
                                             max_memory=max_memory)

print(f'GPU 0: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB')
print(f'GPU 1: {torch.cuda.memory_allocated(1) / 1024**3:.2f} GiB')

print(model.hf_device_map)






