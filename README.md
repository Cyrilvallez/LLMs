# LLMs

This is the main repo containing all the work on code generation by LLMs. It is mostly based on my library [TextWiz](https://github.com/Cyrilvallez/TextWiz) for model inference.

## Install

### Clone the repo

As `TextWiz` is a submodule inside this repository, it is not possible to simply clone this repo the usual way. One also has to initialize the submodule with the `--recurse_submodules` flag: 

```sh
git clone https://github.com/Cyrilvallez/LLMs.git --recurse-submodules
```

One also has to pass the flag whenever pulling upstream changes in order to keep the submodule in sync in case it was also modified:

```sh
git pull --recurse-submodules
```

Or you can also run the following command once:

```sh
git config submodule.recurse true
```

And it will automatically pass the flag whenever you simply `git pull` from the remote.

### Download results

Due to the file size, the results are not included in this repository. To download them, run the following:

```sh
cd LLMs
python3 download_results.py
```

and everything will be added in the correct location.

## Python environment

In case you need to install Conda and are on **linux**, you can run

```sh
source config.sh
```

which will install [mini-forge](https://github.com/conda-forge/miniforge), and create the required environment. In case you already have Conda installed, simply run:

```sh
conda env create -f requirements.yaml
```

to create the computing environment.

## Add models to the library

To add models to the library, one first has to add the appropriate `model.py` file in `TextWiz/textwiz/configs/causal` folder. For examples of what the file should contain, please refer to existing files in the repository - the structure is quite sttraightforward (you need to specify model name, default dtype, number of parameters, choose a family name for the model, default context size, and optionally some additional arguments/version requirements).  

Then, if one of the models you added requires a specific prompt/chat template, navigate to `TextWiz/textwiz/templates`, and modify both `conversation_template.py` and `prompt_template.py`. In both files, you will find a dictionary mapping(`CONVERSATION_MAPPING` and `PROMPT_MAPPING` respectively) between model names and prompt/conversation template class. You should add your model names to these mapping (creating appropriate class if necesary, see class examples in each file) to use proper templates.

## Add datasets to the library

All datasets in this repository (except one) are formatted as `jsonl` files, that is files with one `json` record per line. Each of these `json` record represents a sample of the dataset, and all have the same keys. To add a dataset, add (or create) the new `jsonl` file representing the data somewhere in the `data` folder. Then, navigate to the `helpers/datasets.py` file, and add a new class corresponding to your dataset in the following way:

```python
class YourDataset(SampleDataset):
    """New dataset!!
    """

    # This is where you added your `jsonl`file
    path: str = os.path.join(utils.DATA_FOLDER, 'new_dataset.jsonl')
    # If your dataset has a key corresponding to an ID for each sample, add it here (otherwise set it to whatever string, such as "" or "None")
    id_key: str = 'task_id'
```

You can then easily manipulate the dataset through a `YourDataset` instance.


## Reproducing benchmark results

### Reproducing model outputs

The entry-points to reproduce the model outputs on the different benchmarks are the following:

- **HumanEval**: &ensp;`human_eval.py`
- **AATK**: &ensp;`AATK.py`
- **AATK-Instruct**: &ensp;`AATK_instruct.py`
- **CyberSecEval**: &ensp;`cyber_sec_eval.py`

Each of those scripts takes a `model` as required argument, as well as different optional arguments depending on the given benchmark (used to specify different variations on a specific benchmark, and/or parameters). To get help about optional arguments, you can run `python benchmark_script --help`, replacing `benchmark_script` with one of the above scripts.  

Each of the above scripts also has a `benchmark_script_wrapper.py` (that is, a python file with the added `_wrapper` at the end), and a `benchmark_script.sh` counter-part (once again, replace `benchmark_script` with the actual names given above). Those 2 files are only used to automate launching the given scripts for a large number of models at the same time on a `slrum` GPU cluster.  

Finally, note that depending on hardware, even using a fixed seed may result in different outputs when auto-regresively generating tokens, so results may be slightly different.

### Processing raw code outputs

Once the model code outputs are generated by a model on a given benchmark, you still need to usually parse this code, assert that it is valid (i.e. respect a given programming language syntax), and sometimes run it against unit-tests. To this end, the following scripts will take care of everything:

- **HumanEval**: &ensp;`code_eval.sh` (run it as `bash code_eval.sh`)  
Note that it requires `docker` to build a proper sandbox to run arbitrary code in a properly isolated environment. You can find instructions on how to install it [here](https://docs.docker.com/get-started/get-docker/). `docker` commands need to be usable without `sudo` privileges.
- **AATK** & **AATK-Instruct**: &ensp;`AATK_evaluate.py`  
Note that it requires `CodeQL` static code analyzer to evaluate code. You can find instructions on how to install it [here](https://docs.github.com/en/code-security/codeql-cli/using-the-advanced-functionality-of-the-codeql-cli/advanced-setup-of-the-codeql-cli). You need to follow the instruction on how to add `codeql` to your `PATH` as well.
- **CyberSecEval**: &ensp;`cyber_sec_evaluate.py`

Running each of these scripts will grab the model outputs created in the [previous section](#reproducing-model-outputs), process them accordingly, and save the results of this processing in the correct location.

### Recreating tables and figures

Each benchmark comes with a specific helper file in which you can find the functions that were used to generate tables and figures:

- **HumanEval**: &ensp;`helpers/humaneval.py`
- **AATK** & **AATK-Instruct**: &ensp;`helpers/aatk.py`  
- **CyberSecEval**: &ensp;`helpers/cybersec.py`

They contain specific functions to process the output files created [above](#processing-raw-code-outputs) and compute the required quantities/scores.

## Other scripts

Here is a quick list and description of the remaining scripts.

### Webapps

The three `webapp.py`, `webapp_chat.py` and `experiment_webapp.py` are used to launch simple `gradio` webapps to interact with models in a browser. The first one will allow you to switch model interactively, while the second one is a simplified version, specifically for chat-models. The last one was used for the WalliserDeutsch (strong and weak attackers) experiment.

### Training 

`train.py` and `train_peft.py` are the scripts used to perform the very basic and simple fine-tuning of Llama3 on the WalliserDeutsch dataset (full weights, and peft adapter respectively).

### Reformulations & perplexity

The script `reformulations.py` was used to create the prompt variations of the AATK-Instruct dataset for different models. In turn, `perplexity.py` was used to save the perplexity of each of those prompts with a given reference model to compute PE scores and ME scores.

### Python code extractor

Finally, the file `python_extractor.py` is designed to parse Python code blocks from text files (model outputs).