from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
import torch
import warnings
import time

import engine

# model_name = 'bloom-7.1B'
# model_name = 'stable-lm-7B'
# model_name = 'gpt-j-6B'
# model_name = 'opt-6.7B'
model_name = 'codegen-6B'
# model_name = 'vicuna-7B'
model = engine.HFModel(model_name)

print(f'Memory of the model: {(torch.cuda.max_memory_allocated(0) / 1024**3):.5f} GiB')
torch.cuda.reset_peak_memory_stats(device=0)

max_new_tokens = 200
num_return_sequences = 208
batch_size = 1

prompt = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations.

One of the defining features of monkeys is their incredible diversity. There are over 260 known species of monkeys, each with its own distinct traits and adaptations. They come in a wide range of sizes, from the tiny pygmy marmoset, which can fit in the palm of your hand, to the large and powerful mandrill, known for its strikingly colorful face. This diversity allows monkeys to occupy various ecological niches and adapt to different habitats and diets.

Monkeys are highly social animals, living in complex social structures. They form troops or bands that can range in size from a few individuals to several hundred members, depending on the species. Within these groups, monkeys establish hierarchies through social interactions, with dominant individuals enjoying certain privileges and responsibilities. Social bonds are crucial for their survival, as they provide protection from predators and facilitate cooperative behaviors, such as foraging and caring for young ones.

Another remarkable aspect of monkeys is their exceptional cognitive abilities. They exhibit problem-solving skills, tool usage, and the ability to learn from each other. For instance, certain species of monkeys have been observed using rocks to crack open nuts or sticks to fish for termites. They demonstrate an understanding of cause-and-effect relationships and exhibit a sense of self-awareness. Researchers have conducted numerous studies to explore the cognitive abilities of monkeys, revealing their impressive intellectual capacities.
"""

t0 = time.time()
input_ids = model.tokenizer.encode(prompt, return_tensors='pt').cuda(0)
input_ids, _ = model.model._expand_inputs_for_generation(expand_size=batch_size, input_ids=input_ids)
if 'bloom' in model_name:
    past_key_values = model.model.transformer(input_ids[:, :-1], return_dict=True).past_key_values
elif 'stable-lm' in model_name:
    past_key_values = model.model.gpt_neox(input_ids[:, :-1], return_dict=True).past_key_values
elif 'gpt-j' in model_name:
    past_key_values = model.model.transformer(input_ids[:, :-1], return_dict=True).past_key_values
elif 'opt' in model_name:
    past_key_values = model.model.model.decoder(input_ids[:, :-1], return_dict=True).past_key_values
elif 'codegen' in model_name:
    past_key_values = model.model.transformer(input_ids[:, :-1], return_dict=True).past_key_values
elif 'vicuna' in model_name:
    past_key_values = model.model.model(input_ids[:, :-1], return_dict=True).past_key_values


mem = 0
for i in range(len(past_key_values)):
    for j in range(len(past_key_values[i])):
        mem += past_key_values[i][j].nelement() * past_key_values[i][j].element_size()

print(f'Memory peak: {(torch.cuda.max_memory_allocated(0) / 1024**3):.5f} GiB')
print(f'Memory of keys: {(mem / 1024**3):.2f} GiB')

torch.cuda.reset_peak_memory_stats(device=0)

out2 = model(prompt, num_return_sequences=batch_size, max_new_tokens=2, seed=1,
             batch_size=batch_size, past_key_values=past_key_values)

print(f'Memory peak: {(torch.cuda.max_memory_allocated(0) / 1024**3):.5f} GiB')

# out1 = model(prompt, num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens,
#              seed=1, batch_size=batch_size, past_key_values=past_key_values)
# dt = time.time() - t0
# print(f'Time with precomputed hidden states: {dt:.2f} s')
# print(f'Memory peak with precomputed states: {(torch.cuda.max_memory_allocated(0) / 1024**3):.5f} GiB')


# del model
# torch.cuda.reset_peak_memory_stats(device=0)
# model = engine.HFModel(model_name)


# t1 = time.time()
# out2 = model(prompt, num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens, seed=1,
#              batch_size=batch_size)
# dt1 = time.time() - t1
# print(f'Time withOUT precomputed hidden states: {dt:.2f} s')
# print(f'Memory peak withOUT precomputed states: {(torch.cuda.max_memory_allocated(0) / 1024**3):.5f} GiB')

# print(f'Same results: {out1 == out2}')