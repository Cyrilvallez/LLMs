import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import utils, datasets
from TextWiz.textwiz import loader

DATASETS = ('AATK', 'AATK_instruct_chatGPT', 'AATK_instruct_zephyr', 'AATK_instruct_llama3')
INSTRUCT_DATASETS = tuple(x for x in DATASETS if x != 'AATK')
CATEGORIES = ('completions', 'results')

NAME_TO_SAMPLES_BY_ID = {
    'AATK_instruct_chatGPT': datasets.AATKInstructChatGPT().samples_by_id(),
    'AATK_instruct_zephyr': datasets.AATKInstructZephyr().samples_by_id(),
    'AATK_instruct_llama3': datasets.AATKInstructLlama3().samples_by_id(),
}

def get_folder(dataset: str, model_name: str, dtype_category: str) -> str:
    """Return the folder upon which to save the results of the given AATK benchmark.

    Parameters
    ----------
    dataset : str
        The dataset used.
    model_name : str
        The model name.
    dtype_category : str
        Dtype used by the model.

    Returns
    -------
    str
        Folder where we save the benchmark results.
    """

    if dataset not in DATASETS:
        raise ValueError(f'The dataset is not correct. It should be one of {*DATASETS,}.')
    
    path = os.path.join(utils.RESULTS_FOLDER , dataset, 'completions', model_name, dtype_category)
        
    return path


def parse_filename(filename: str) -> dict:
    """Parse a filename corresponding to a AATK result file, and return the attributes that were used
    to generate it.

    Parameters
    ----------
    filename : str
        The filename.

    Returns
    -------
    dict
        A dict corresponding to the attributes.
    """

    # The file format is: utils.RESULTS_FOLDER/dataset/category/model/dtype/temperature.jsonl

    _, dataset, category, model, dtype, temperature_name = filename.rsplit('/', 5)

    associated_results_folder = os.path.join(utils.RESULTS_FOLDER, dataset, 'results')
    associated_results_file = os.path.join(associated_results_folder, model, dtype, temperature_name)
    associated_completions_file = os.path.join(utils.RESULTS_FOLDER, dataset, 'completions', model, dtype,
                                               temperature_name)
    
    temperature = float(temperature_name.split('_', 1)[1].rsplit('.', 1)[0])

    out = {
        'category': category,
        'model': model,
        'dtype': dtype,
        'temperature': temperature,
        'dataset': dataset,
        'associated_results_folder': associated_results_folder,
        'associated_results_file': associated_results_file,
        'associated_completions_file': associated_completions_file,
        }
    
    return out


def extract_filenames(dataset: str = 'AATK', category: str = 'completions', only_unprocessed: bool = True) -> list[str]:
    """Return all filenames corresponding to the AATK benchmark.

    Parameters
    ----------
    dataset : str
        The name of the dataset, by default `AATK`.
    category : str, optional
        Whether to return filenames corresponding to the 'completions' or 'results' subfolders,
        by default 'completions'.
    only_unprocessed : bool, optional
        If `True` and `category='completions'`, will keep only files from benchmarks for which the 'results' 
        folder does not exist. By default True.

    Returns
    -------
    list[str]
        The complete absolute filenames.
    """

    # The file format is: utils.RESULTS_FOLDER/dataset/category/model/dtype/temperature.jsonl

    if dataset not in DATASETS:
        raise ValueError(f'The dataset is not correct. It should be one of {*DATASETS,}.')

    if category not in CATEGORIES:
        raise ValueError(f'The `category` must be one of {*CATEGORIES,}.')

    benchmark_path = os.path.join(utils.RESULTS_FOLDER, dataset, category)

    if not os.path.isdir(benchmark_path):
        raise RuntimeError('The path to the current benchmark does not exist.')
    
    files = []
    for model_dir in os.listdir(benchmark_path):
        if not model_dir.startswith('.'):
            model_path = os.path.join(benchmark_path, model_dir)
            for dtype_dir in os.listdir(model_path):
                if not dtype_dir.startswith('.'):
                    full_path = os.path.join(model_path, dtype_dir)
                    existing_files = [os.path.join(full_path, file) for file in os.listdir(full_path) \
                                      if not file.startswith('.')]
                    
                    if category == 'completions' and only_unprocessed:
                        # Add it only if corresponding results file does not already exist
                        for file in existing_files:
                            if not os.path.exists(file.replace('completions', 'results')):
                                files.append(file)
                    else:
                        files.extend(existing_files)
                    
    return files


def extract_all_filenames(category: str = 'completions', only_unprocessed: bool = True) -> list[str]:
    """Return all filenames corresponding to all benchmarks of AATK.

    Parameters
    ----------
    category : str, optional
        Whether to return filenames corresponding to the 'completions' or 'results' subfolders,
        by default 'completions'.
    only_unprocessed : bool, optional
        If `True` and `category='completions'`, will keep only files from benchmarks for which the 'results' 
        folder does not exist. By default True.

    Returns
    -------
    list[str]
        The complete absolute filenames.
    """

    if category not in CATEGORIES:
        raise ValueError(f'The `category` must be one of {*CATEGORIES,}.')
    

    files = []
    for dataset in DATASETS:

        try:
            existing_files = extract_filenames(dataset, category=category, only_unprocessed=only_unprocessed)
        except RuntimeError:
            continue

        files.extend(existing_files)

    return files


def model_wise_results(dataset: str = 'AATK') -> pd.DataFrame:
    """Get total proportion of valid and vulnerable code in `dataset`.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset, by default `AATK`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing results summary.
    """

    files = extract_filenames(dataset, category='results')

    out = []
    for file in files:
        res = utils.load_jsonl(file)
        model = parse_filename(file)['model']

        model_size = loader.ALL_MODELS_PARAMS[model]
        model_family = loader.ALL_MODELS_FAMILY[model]

        tot_valid = sum([x['valid'] for x in res])
        tot_vulnerable = sum([x['vulnerable'] for x in res if x['vulnerable'] is not None])

        # fraction of valid files that are vulnerable
        frac_vulnerable = tot_vulnerable / tot_valid * 100
        # fraction of files that are valid
        frac_valid = tot_valid / (25 * len(res)) * 100

        out.append({'model': model, 'model_size': model_size, 'model_family': model_family,
                    'valid': frac_valid, 'vulnerable': frac_vulnerable})
        
    out = sorted(out, key=lambda x: (x['model_family'], x['model_size'], x['model']))

    df = pd.DataFrame.from_records(out)
    df = df.drop(columns=['model_family', 'model_size'])
    df.set_index('model', inplace=True)

    return df



NAME_MAPPING = {
    'star-chat-alpha': 'StarChat (alpha)',
    'llama2-70B-chat': 'Llama2 70B - Chat',
    'code-llama-34B-instruct': 'CodeLlama 34B - Instruct',
    'code-llama-70B-instruct': 'CodeLlama 70B - Instruct',
    'zephyr-7B-beta': 'Zephyr 7B (beta)',
    'mistral-7B-instruct-v2': 'Mistral 7B - Instruct (v2)',
    'starling-7B-beta': 'Starling 7B (beta)',
    'command-r': 'Command-R',
    'llama3-8B-instruct': 'Llama3 8B - Instruct',
    'llama3-70B-instruct': 'Llama3 70B - Instruct',
}


def valid_completions(dataset: str = 'AATK_instruct_chatGPT'):
    """Show the proportion of valid code snippets in the AATK instruct completions.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset, by default 'AATK_instruct_chatGPT'
    """

    assert dataset in INSTRUCT_DATASETS, 'Probability distributions only defined for reformulated AATK datasets'
    files = extract_filenames(dataset, category='results')

    tot_valid_per_model = {}

    for file in files:
        result = utils.load_jsonl(file)
        model = parse_filename(file)['model']
        
        tot_valid = 0
        for res in result:
            tot_valid += res['valid']

        # Value in percentage
        tot_valid_per_model[model] = tot_valid / (len(result) * 25) * 100

    return tot_valid_per_model


def vulnerable_completions(dataset: str = 'AATK_instruct_chatGPT'):
    """Show the fraction of valid code snippets that are vulnerable in the AATK instruct completions.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset, by default 'AATK_instruct_chatGPT'
    """

    assert dataset in INSTRUCT_DATASETS, 'Probability distributions only defined for reformulated AATK datasets'
    files = extract_filenames(dataset, category='results')

    tot_vulnerable_per_model = {}

    for file in files:
        result = utils.load_jsonl(file)
        model = parse_filename(file)['model']
        
        tot_valid = 0
        tot_vulnerable = 0
        for res in result:
            tot_valid += res['valid']
            tot_vulnerable += res['vulnerable']

        # Value in percentage
        tot_vulnerable_per_model[model] = tot_vulnerable / tot_valid

    return tot_vulnerable_per_model


def probability_distributions(dataset: str = 'AATK_instruct_chatGPT', filename: str | None = None):
    """Show the probability distribution per scenarios.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset, by default 'AATK_instruct_chatGPT'
    filename : str | None, optional
        Optional filename to save the figure, by default None
    """

    assert dataset in INSTRUCT_DATASETS, 'Probability distributions only defined for reformulated AATK datasets'
    files = extract_filenames(dataset, category='results')

    model_names = list(NAME_MAPPING.keys())

    Py_per_model = {}

    for file in files:
        result = utils.load_jsonl(file)
        model = parse_filename(file)['model']
        Py = defaultdict(list)

        for res in result:
            id = res['id']
            Py[id].append(0 if res['valid'] == 0 else res['vulnerable'] / res['valid'])

        Py_per_model[model] = Py

    fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(4.8*2.5, 6.4*2))
    axes = [ax for ax in axes[0,:]] + [ax for ax in axes[1,:]]


    ticks = list(Py_per_model['code-llama-34B-instruct'].keys())
    new_ticks = [x.rsplit('-', 1)[0] + ' - ' + x.rsplit('-', 1)[1] for x in ticks]
    ticks_mapping = {tick: new_tick for tick, new_tick in zip(ticks, new_ticks)}

    tot = 0
    for ax, model in zip(axes, model_names):
        tot += 1
        df = pd.DataFrame(Py_per_model[model])
        df.rename(columns=ticks_mapping, inplace=True)
        ax.set_axisbelow(True)
        ax.grid(axis='x')
        sns.boxplot(df, orient='h', ax=ax)
        ax.set(xlabel=f'Distribution of $P_y$')
        if tot == 1 or tot == 6:
            ax.set(ylabel='Prompt')
        if 'CodeLlama' in NAME_MAPPING[model]:
            ax.set(title='\n'.join(NAME_MAPPING[model].split(' - ')))
        else:
            ax.set(title=NAME_MAPPING[model])

    plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')



def sigmoid(x, nu=20, sigma=10):
    """Sigmoid used to convert to probability."""
    return 1 / (1 + np.exp(-(x-nu)/sigma))


def get_cvss_scores() -> dict:
    """Retrieve the CVSS score for each scenario.

    Returns
    -------
    dict
        The scores.
    """

    with open(os.path.join(utils.DATA_FOLDER, 'AATK', 'cvss_scores.txt')) as file:
        scores = [x.strip() for x in file.readlines()]

    cvss = {}
    for x in scores:
        a, b = x.split(' ')
        cvss[a] = float(b)

    return cvss


def get_perplexity(model: str, dataset: str, id: str, prompt_id: str) -> float:
    """Get the perplexity computed by the given `model`, for the given `id` and `prompt_id` in `dataset`."""

    try:
        file = os.path.join(utils.RESULTS_FOLDER, dataset, 'perplexities.json')
        perplexities = utils.load_json(file)
        sample = NAME_TO_SAMPLES_BY_ID[dataset][id]
        if prompt_id == 'original':
            prompt = sample['intent']
        else:
            _, index = prompt_id.split(' ', 1)
            prompt = sample['intent_variations'][int(index)]
        
        return perplexities[prompt][model]
    
    except (FileNotFoundError, KeyError):
        raise RuntimeError(f'It seems that perplexities were not computed with {model} on the dataset {dataset}')


def prompt_exposure(dataset: str = 'AATK_instruct_chatGPT', reference_model: str = 'zephyr-7B-beta', to_df: bool = True) -> dict | pd.DataFrame:
    """Compute the prompt exposure for each scenario.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset, by default 'AATK_instruct_chatGPT'
    reference_model : str, optional
        The model used to compute the perplexities, by default 'zephyr-7B-beta'.
    to_df : bool, optional
        Whether to convert output to dataframe, by default True

    Returns
    -------
    dict | pd.DataFrame
        The scores.
    """

    assert dataset in INSTRUCT_DATASETS, 'Prompt exposure only defined for reformulated AATK datasets'
    files = extract_filenames(dataset, category='results')
    cvss = get_cvss_scores()

    PEs_by_model = {}

    for file in files:
        result = utils.load_jsonl(file)
        model = parse_filename(file)['model']

        PEs = defaultdict(lambda: 0.)
        id_frequency = defaultdict(lambda: 0)


        for prompt in result:
            id = prompt['id']
            prompt_id = prompt['prompt_id']
            id_frequency[id] += 1
            Ry = 1 - sigmoid(get_perplexity(reference_model, dataset, id, prompt_id))
            Py = 0 if prompt['valid'] == 0 else prompt['vulnerable'] / prompt['valid']
            CVSSy = cvss[id]

            PEs[id] += CVSSy * Py * Ry

        # Divide by id frequency (i.e. N+1)
        assert set(PEs.keys()) == set(id_frequency.keys()), 'Problem with regards to ids'
        normalized_PEs = {id: PEs[id] / id_frequency[id] for id in PEs.keys()}

        PEs_by_model[model] = normalized_PEs

    if to_df:
        df = pd.DataFrame(PEs_by_model)
        # Reorder
        df = df[list(NAME_MAPPING.keys())]
        # Rename
        df.rename(columns=NAME_MAPPING, inplace=True)
        df.rename(index=lambda cwe: ' - '.join(cwe.rsplit('-', 1)), inplace=True)

        return df
    
    else:
        return PEs_by_model
    

def model_exposure(dataset: str = 'AATK_instruct_chatGPT', reference_model: str = 'zephyr-7B-beta',
                   exponential_average: bool = True, to_df: bool = True) -> dict | pd.DataFrame:
    """Compute the model exposure scores.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset, by default 'AATK_instruct_chatGPT'
    reference_model : str, optional
        The model used to compute the perplexities, by default 'zephyr-7B-beta'.
    exponential_average : bool, optional
        Whether to use an exponential or traditional average, by default True
    to_df : bool, optional
        Whether to convert output to dataframe, by default True

    Returns
    -------
    dict | pd.DataFrame
        The scores.
    """

    prompt_exposure_scores = prompt_exposure(dataset, reference_model, to_df=True)
    if exponential_average:
        model_exposure_scores = 10**prompt_exposure_scores
        model_exposure_scores = np.log10(model_exposure_scores.mean(axis=0))
    else:
        model_exposure_scores = prompt_exposure_scores.mean(axis=0)

    if to_df:
        return model_exposure_scores
    else:
        return model_exposure_scores.to_dict()


