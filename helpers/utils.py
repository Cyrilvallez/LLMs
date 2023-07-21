import torch
import numpy as np
import random
import os
import json
import multiprocessing as mp

# Path to the root of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))

# Path to the data folder
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

# Path to the results folder
RESULTS_FOLDER = os.path.join(ROOT_FOLDER, 'results')


# Most frequent text/data file extensions
FREQUENT_EXTENSIONS = (
    'json',
    'jsonl',
    'txt',
    'csv'
)


def set_all_seeds(seed: int):
    """Set seed for all random number generators (random, numpy and torch).

    Parameters
    ----------
    seed : int
        The seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_output(predictions: list[str]) -> str:
    """Format a list of strings corresponding to model predictions into a single string.

    Parameters
    ----------
    predictions : list[str]
        The model predictions.

    Returns
    -------
    str
        Formatted string.
    """

    if len(predictions) == 1:
        return predictions[0]
    else:
        out = f''
        for i, pred in enumerate(predictions):
            out += f'Sequence {i+1}:\n{pred}'
            if i != len(predictions)-1:
                out += '\n\n'
        return out


def validate_filename(filename: str, extension: str = 'json') -> str:
    """Format and check the validity of a filename and its extension. Create the path if needed, and 
    add/manipulate the extension if needed.

    Parameters
    ----------
    filename : str
        The filename to check for.
    extension : str, optional
        The required extension for the filename, by default 'json'

    Returns
    -------
    str
        The filename, reformated if needed.
    """

    # Extensions are always lowercase
    extension = extension.lower()
    # Remove dots in extension if any
    extension = extension.replace('.', '')

    dirname, basename = os.path.split(filename)

    # Check that the extension and basename are correct
    if basename == '':
        raise(ValueError('The basename cannot be empty'))
    
    split_on_dots = basename.split('.')

    # In this case add the extension at the end
    if len(split_on_dots) == 1:
        basename += '.' + extension
    # In this case there may be an extension, and we check that it is the correct one and change it if needed
    else:
        # The extension is correct
        if split_on_dots[-1] == extension:
            pass
        # There is a frequent extension, but not the correct one -> we change it
        elif split_on_dots[-1] in FREQUENT_EXTENSIONS:
            basename = '.'.join(split_on_dots[0:-1]) + '.' + extension
        # We did not detect any extension -> just add it at the end
        else:
            basename = '.'.join(split_on_dots) + '.' + extension

    # Check that the given path goes through the project repository
    dirname = os.path.abspath(dirname)
    if not (dirname.startswith(ROOT_FOLDER + os.sep) or dirname == ROOT_FOLDER):
        raise(ValueError('The path you provided is outside the project repository.'))

    # Make sure the path exists, and creates it if this is not the case
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return os.path.join(dirname, basename)


def save_json(dictionary: dict, filename: str):
    """
    Save a dictionary to disk as a json file.

    Parameters
    ----------
    dictionary : dict
        The dictionary to save.
    filename : str
        Filename to save the file.
    """
    
    filename = validate_filename(filename, extension='json')
    
    with open(filename, 'w') as fp:
        json.dump(dictionary, fp, indent='\t')
        
        
def save_jsonl(dictionaries: list[dict], filename: str, append: bool = False):
    """Save a list of dictionaries to a jsonl file.

    Parameters
    ----------
    dictionaries : list[dict]
        The list of dictionaries to save.
    filename : str
        Filename to save the file.
    append : bool
        Whether to append at the end of the file or create a new one, default to False.
    """

    filename = validate_filename(filename, extension='jsonl')

    mode = 'a' if append else 'w'

    with open(filename, mode) as fp:
        for dic in dictionaries:
            fp.write(json.dumps(dic) + '\n')


def load_json(filename: str) -> dict:
    """
    Load a json file and return a dictionary.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    data : dict
        The dictionary representing the file.

    """
    
    with open(filename, 'r') as fp:
        data = json.load(fp)

    return data


def load_jsonl(filename: str) -> list[dict]:
    """Load a jsonl file as a list of dictionaries.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    list[dict]
        The list of dictionaries.
    """

    dictionaries = []

    with open(filename, 'r') as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                # yield json.loads(line)
                dictionaries.append(json.loads(line))

    return dictionaries


def find_rank_of_subprocess_inside_the_pool():
    """Find the rank of the current subprocess inside the pool that was launched either by 
    multiprocessing.Pool() or concurrent.futures.ProcessPoolExecutor().
    If called from the main process, return 0.
    Note that this is a bit hacky but work correctly because both methods provide the rank of the subprocesses
    inside the subprocesses name as 'SpawnPoolWorker-RANK' or 'SpawnProcess-RANK' respectively.
    """

    process = mp.current_process()

    if process.name == 'MainProcess':
        rank = 0
    elif isinstance(process, mp.context.SpawnProcess) or isinstance(process, mp.context.ForkProcess):
        # Provide rank starting at 0 instead of 1
        try:
            rank = int(process.name[-1]) - 1
        except ValueError:
            raise RuntimeError('Cannot retrieve the rank of the current subprocess.')
    else:
        raise RuntimeError('The type of the running process is unknown.')
        
    return rank


def set_cuda_visible_device(gpu_rank: int | list[int]):
    """Set cuda visible devices to `gpu_rank` only.

    Parameters
    ----------
    gpu_rank : int | list[int]
        The GPUs we want to be visible.
    """

    if type(gpu_rank) == int:
        gpu_rank = [gpu_rank]

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_rank)


def set_cuda_visible_device_of_subprocess():
    """Set the cuda visible device of a subprocess inside a pool to only the gpu with same
    rank as the subprocess.
    """

    gpu_rank = find_rank_of_subprocess_inside_the_pool()
    set_cuda_visible_device(gpu_rank)