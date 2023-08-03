import torch
import os
from concurrent.futures import ProcessPoolExecutor
import argparse
import gc

import engine
from helpers import utils


large_text = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations.

One of the defining features of monkeys is their incredible diversity. There are over 260 known species of monkeys, each with its own distinct traits and adaptations. They come in a wide range of sizes, from the tiny pygmy marmoset, which can fit in the palm of your hand, to the large and powerful mandrill, known for its strikingly colorful face. This diversity allows monkeys to occupy various ecological niches and adapt to different habitats and diets.

Monkeys are highly social animals, living in complex social structures. They form troops or bands that can range in size from a few individuals to several hundred members, depending on the species. Within these groups, monkeys establish hierarchies through social interactions, with dominant individuals enjoying certain privileges and responsibilities. Social bonds are crucial for their survival, as they provide protection from predators and facilitate cooperative behaviors, such as foraging and caring for young ones.

Another remarkable aspect of monkeys is their exceptional cognitive abilities. They exhibit problem-solving skills, tool usage, and the ability to learn from each other. For instance, certain species of monkeys have been observed using rocks to crack open nuts or sticks to fish for termites. They demonstrate an understanding of cause-and-effect relationships and exhibit a sense of self-awareness. Researchers have conducted numerous studies to explore the cognitive abilities of monkeys, revealing their impressive intellectual capacities.

Monkeys are primarily herbivorous but have a diverse diet that includes fruits, leaves, seeds, and insects. Some species, like the howler monkey, are specialized folivores, consuming mainly leaves to meet their nutritional needs. Others, such as the capuchin monkey, are known for their omnivorous diet, which includes fruits, nuts, insects, and even small vertebrates. Their varied diet contributes to the dispersal of seeds, making monkeys important agents in forest regeneration and maintaining biodiversity.

Monkeys play a crucial role in their ecosystems. As both predators and prey, they contribute to the balance of their habitats. They aid in seed dispersal, pollination, and nutrient cycling, thereby influencing the structure and dynamics of plant communities. Additionally, monkeys are indicators of ecosystem health, as their presence or absence can reflect the overall well-being of an ecosystem.

Despite their significance, monkeys face numerous challenges and threats. Habitat loss due to deforestation, fragmentation, and human encroachment is one of the primary concerns. Additionally, illegal wildlife trade and hunting pose significant risks to monkey populations. Conservation efforts, including protected areas and education campaigns, are vital to ensure the survival of these remarkable creatures.

In conclusion, monkeys are extraordinary creatures that captivate us with their diversity, social structures, cognitive abilities, and ecological importance. Their lives are intricately woven into the tapestry of their respective habitats, and understanding and protecting them is crucial for maintaining the balance of our planet's ecosystems. By appreciating and conserving these fascinating animals, we can continue to learn from them and be inspired by their remarkable qualities.
"""

SMALL_MODELS = (
    'bloom-560M',
    'bloom-1.7B',
    'bloom-3B',
    'bloom-7.1B',
    'stable-lm-3B',
    'stable-lm-7B',
    'star-coder-base',
    'star-coder',
    'star-coder-plus',
    'star-chat-alpha',
    'star-chat-beta',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'gpt-j-6B',
    'gpt-neo-125M',
    'gpt-neo-1.3B',
    'gpt-neo-2.7B',
    'opt-125M',
    'opt-350M',
    'opt-1.3B',
    'opt-2.7B',
    'opt-6.7B',
    'opt-13B',
    'codegen-350M',
    'codegen-2B',
    'codegen-6B',
    'codegen-16B',
    'codegen2-1B',
    'codegen2-3.7B',
    'codegen2-7B',
    'codegen2-16B',
    'codegen25-7B',
    'codegen25-7B-instruct',
    'vicuna-7B',
    'vicuna-13B',
)

input_sizes = [50*i for i in range(1, 11)]
max_tokens = [50*i for i in range(1, 11)]
max_tokens += [512]

def memory_estimation(model_name: str):

    model = engine.HFModel(model_name)
    # model_memory = torch.cuda.max_memory_allocated() / 1024**3
    model_memory = model.max_memory_footprint
    large_tokens = model.tokenizer.encode(large_text, return_tensors='pt')
    model_memory_consumption = {}

    for i, input_size in enumerate(input_sizes):

        prompt = model.tokenizer.batch_decode(large_tokens[:, :input_size], skip_special_tokens=True)[0]
        input_size_memory_consumption = {}

        for j, max_token in enumerate(max_tokens):

            torch.cuda.reset_peak_memory_stats()
            actual_peak = torch.cuda.max_memory_allocated() / 1024**3

            foo = model(prompt, num_return_sequences=1, max_new_tokens=max_token, batch_size=1)
            # mem = torch.cuda.max_memory_allocated() / 1024**3 - model_memory
            mem = (torch.cuda.max_memory_allocated() / 1024**3) - actual_peak
            input_size_memory_consumption[max_token] = mem

        model_memory_consumption[input_size] = input_size_memory_consumption

    filename = os.path.join(utils.ROOT_FOLDER, 'memory_estimator2', f'{model_name}.json')
    utils.save_json(model_memory_consumption, filename)

    del model
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Memory estimation')
    parser.add_argument('--gpus', type=int, default=5, help='The number of GPUs to use.')
    
    args = parser.parse_args()
    num_gpus = args.gpus

    with ProcessPoolExecutor(max_workers=num_gpus, initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
        pool.map(memory_estimation, SMALL_MODELS, chunksize=1)