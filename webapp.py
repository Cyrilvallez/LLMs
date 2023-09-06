import gc
import os

import torch
import gradio as gr

import engine
from engine import loader
from helpers import utils


# Default model to load at start-up
DEFAULT = 'llama2-7B-chat' if torch.cuda.is_available() else 'bloom-560M'

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# Initialize global model (necessary not to reload the model for each new inference)
model = engine.HFModel(DEFAULT)

# Initialize a global conversation object for chatting with the models
conversation = model.get_empty_conversation()


def update_model(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False):
    """Update the model and conversation in the global scope so that we can reuse them and speed up inference.

    Parameters
    ----------
    model_name : str
        The name of the new model to use.
    quantization : bool, optional
        Whether to load the model in 8 bits mode.
    """
    
    global model
    global conversation

    if quantization_8bits and quantization_4bits:
        raise gr.Error('You cannot use both int8 and int4 quantization. Choose either one and try reloading.')
    
    if (quantization_8bits or quantization_4bits) and not torch.cuda.is_available():
        raise gr.Error('You cannot use quantization if you run without GPUs.')

    # Delete the variables if they exist (they should except if there was an error when loading a model at some point)
    # to save memory before loading the new one
    try:
        del model
        del conversation
        gc.collect()
    except NameError:
        pass

    # Try loading the model
    try:
        model = engine.HFModel(model_name, quantization_8bits=quantization_8bits,
                               quantization_4bits=quantization_4bits)
        conversation = model.get_empty_conversation()
    except:
        raise gr.Error('There was an error loading this model. Please retry or choose another one.')
    
    # Return values to clear the input and output textboxes, and input and output chatbot boxes
    return '', '', '', [[None, None]]



def text_generation(prompt: str, max_new_tokens: int = 60, do_sample: bool = True, top_k: int = 50,
                    top_p: float = 0.90, temperature: float = 0.9, num_return_sequences: int = 1,
                    use_seed: bool = False, seed: int | None = None) -> str:
    """Text generation using the model and tokenizer in the global scope, so that we can reuse them for multiple
    prompts.

    Parameters
    ----------
    prompt : str
        The prompt to the model.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.
    do_sample : bool, optional
        Whether to introduce randomness in the generation, by default True.
    top_k : int, optional
        How many tokens with max probability to consider for randomness, by default 50.
    top_p : float, optional
        The probability density covering the new tokens to consider for randomness, by default 0.9.
    temperature : float, optional
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness), by default 0.9.
    num_return_sequences : int, optional
        How many sequences to generate according to the `prompt`, by default 1.
    use_seed : bool, optional
        Whether to use a fixed seed for reproducibility, by default False.
    seed : Union[None, int], optional
        An optional seed to force the generation to be reproducible.
    Returns
    -------
    str
        String containing all `num_return_sequences` sequences generated.
    """
    
    if not use_seed:
        seed = None

    try:
        predictions = model(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                            temperature=temperature, num_return_sequences=num_return_sequences, seed=seed,
                            truncate_prompt_from_output=True)
    except BaseException as e:
        raise gr.Error(f'The following error happened during generation: {repr(e)}')
    
    if num_return_sequences > 1:
        return utils.format_output(predictions)
    else:
        return predictions



def chat_generation(prompt: str, max_new_tokens: int = 60, do_sample: bool = True, top_k: int = 40,
                    top_p: float = 0.90, temperature: float = 0.9, use_seed: bool = False,
                    seed: int | None = None) -> tuple[str, list[tuple[str, str]]]:
    """Chat generation using the model, tokenizer and conversation in the global scope, so that we can reuse
    them for multiple prompts.

    Parameters
    ----------
    prompt : str
        The prompt to the model.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.
    do_sample : bool, optional
        Whether to introduce randomness in the generation, by default True.
    top_k : int, optional
        How many tokens with max probability to consider for randomness, by default 100.
    top_p : float, optional
        The probability density covering the new tokens to consider for randomness, by default 0.92.
    temperature : float, optional
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness), by default 0.9.
    use_seed : bool, optional
        Whether to use a fixed seed for reproducibility., by default False.
    seed : Union[None, int], optional
        An optional seed to force the generation to be reproducible.

    Returns
    -------
    tuple[str, list[tuple[str, str]]]
        An empty string to reinitialize the prompt box, and the conversation as a list[tuple[str, str]].
    """
    
    if not use_seed:
        seed = None
    
    # This will update the global conversation in-place
    try:
        _ = model.generate_conversation(prompt, conv_history=conversation, max_new_tokens=max_new_tokens,
                                        do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                        seed=seed)
    except BaseException as e:
        raise gr.Error(f'The following error happened during generation: {repr(e)}')
    
    # The first output is an empty string to clear the input box, the second is the format output to use in a
    # gradio chatbot component
    return '', conversation.to_gradio_format()



def authentication(username: str, password: str) -> bool:
    """Simple authentication method.

    Parameters
    ----------
    username : str
        The username provided.
    password : str
        The password provided.

    Returns
    -------
    bool
        Return True if both the username and password match the credentials stored in `CREDENTIALS_FILE`. 
        False otherwise.
    """

    with open(CREDENTIALS_FILE, 'r') as file:
        # Read lines and remove whitespaces
        lines = [line.strip() for line in file.readlines()]

    valid_username = lines[0]
    valid_password = lines[1]

    if (username == valid_username) & (password == valid_password):
        return True
    else:
        return False
    

def clear_chatbot():
    """Erase the conversation history and reinitialize the elements.
    """

    # Erase the conversation history before
    conversation.erase_conversation()
    return '', conversation.to_gradio_format()
    


# Define general elements of the UI (generation parameters)
model_name = gr.Dropdown(loader.ALLOWED_MODELS, value=DEFAULT, label='Model name',
                         info='Choose the model you want to use.', multiselect=False)
quantization_8bits = gr.Checkbox(value=False, label='int8 quantization', visible=torch.cuda.is_available())
quantization_4bits = gr.Checkbox(value=False, label='int4 quantization', visible=torch.cuda.is_available())
max_new_tokens = gr.Slider(10, 1000, value=250, step=10, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
do_sample = gr.Checkbox(value=True, label='Sampling', info=('Whether to incorporate randomness in generation. '
                                                            'If not selected, perform greedy search.'))
top_k = gr.Slider(0, 200, value=50, step=5, label='Top-k',
               info='How many tokens with max probability to consider.')
top_p = gr.Slider(0, 1, value=0.90, step=0.01, label='Top-p',
              info='Probability density threshold for new tokens.')
temperature = gr.Slider(0, 1, value=0.9, step=0.01, label='Temperature',
                        info='How to cool down the probability distribution.')
num_return_sequence = gr.Slider(1, 10, value=1, step=1, label='Sequence', info='Number of sequence to generate.')
use_seed = gr.Checkbox(value=False, label='Use seed', info='Whether to use a fixed seed for reproducibility.')
seed = gr.Number(0, label='Seed', info='Seed for reproducibility.', precision=0)
load_button = gr.Button('Load model', variant='primary')

# Define elements of the simple generation Tab
prompt_text = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
output_text = gr.Textbox(label='Model output')
generate_button_text = gr.Button('Generate text', variant='primary')
clear_button_text = gr.Button('Clear prompt', variant='secondary')

# Define elements of the chatbot Tab
prompt_chat = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
output_chat = gr.Chatbot(label='Conversation')
generate_button_chat = gr.Button('Generate text', variant='primary')
clear_button_chat = gr.Button('Clear conversation')

# Define the inputs for the main inference
inputs_to_simple_generation = [prompt_text, max_new_tokens, do_sample, top_k, top_p, temperature,
                               num_return_sequence, use_seed, seed]
inputs_to_chatbot = [prompt_chat, max_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]

# Define inputs for the logging callbacks
inputs_to_text_callback = [model_name, quantization_8bits, quantization_4bits, *inputs_to_simple_generation,
                           output_text]
inputs_to_chat_callback = [model_name, quantization_8bits, quantization_4bits, *inputs_to_chatbot,
                           output_chat]

# set-up callbacks for flagging and automatic logging
automatic_logging_text = gr.CSVLogger()
automatic_logging_chat = gr.CSVLogger()

# Some prompt examples
prompt_examples = [
    "Please write a function to multiply 2 numbers `a` and `b` in Python.",
    "Hello, what's your name?",
    "What's the meaning of life?",
    "How can I write a Python function to generate the nth Fibonacci number?",
    ("Here is my data {'Name':['Tom', 'Brad', 'Kyle', 'Jerry'], 'Age':[20, 21, 19, 18], 'Height' :"
     " [6.1, 5.9, 6.0, 6.1]}. Can you provide Python code to plot a bar graph showing the height of each person?"),
]


demo = gr.Blocks(title='Text generation with LLMs')

with demo:

    # Need to wrap everything in a row because we want two side-by-side columns
    with gr.Row():

        # First column where we have prompts and outputs. We use large scale because we want a 1.7:1 ratio
        # but scale needs to be an integer
        with gr.Column(scale=17):

            # Tab 1 for simple text generation
            with gr.Tab('Text generation'):
                prompt_text.render()
                with gr.Row():
                    generate_button_text.render()
                    clear_button_text.render()
                output_text.render()

                gr.Markdown("### Prompt Examples")
                gr.Examples(prompt_examples, inputs=prompt_text)

            # Tab 2 for chat mode
            with gr.Tab('Chat mode'):
                prompt_chat.render()
                with gr.Row():
                    generate_button_chat.render()
                    clear_button_chat.render()
                output_chat.render()

                gr.Markdown("### Prompt Examples")
                gr.Examples(prompt_examples, inputs=prompt_chat)

        # Second column defines model selection and generation parameters
        with gr.Column(scale=10):
                
            # First box for model selection
            with gr.Box():
                gr.Markdown("### Model selection")
                with gr.Row():
                    model_name.render()
                with gr.Row():
                    quantization_8bits.render()
                    quantization_4bits.render()
                with gr.Row():
                    load_button.render()
            
            # Accordion for generation parameters
            with gr.Accordion("Text generation parameters", open=False):
                with gr.Row():
                    do_sample.render()
                    max_new_tokens.render()
                with gr.Row():
                    top_k.render()
                    top_p.render()
                with gr.Row():
                    temperature.render()
                    num_return_sequence.render()
                with gr.Row():
                    use_seed.render()
                    seed.render()

    # Perform simple text generation when clicking the button
    generate_event1 = generate_button_text.click(text_generation, inputs=inputs_to_simple_generation,
                                                 outputs=output_text)
    # Add automatic callback on success
    generate_event1.success(lambda *args: automatic_logging_text.flag(args), inputs=inputs_to_text_callback,
                            preprocess=False)

    # Perform chat generation when clicking the button
    generate_event2 = generate_button_chat.click(chat_generation, inputs=inputs_to_chatbot,
                                                 outputs=[prompt_chat, output_chat])
    # Add automatic callback on success
    generate_event2.success(lambda *args: automatic_logging_chat.flag(args), inputs=inputs_to_chat_callback,
                            preprocess=False)

    # Switch the model loaded in memory when clicking
    events_to_cancel = [generate_event1, generate_event2]
    load_button.click(update_model, inputs=[model_name, quantization_8bits, quantization_4bits],
                      outputs=[prompt_text, output_text, prompt_chat, output_chat], cancels=events_to_cancel)
    
    # Clear the prompt and output boxes when clicking the button
    clear_button_text.click(lambda: ['', ''], outputs=[prompt_text, output_text])
    clear_button_chat.click(clear_chatbot, outputs=[prompt_chat, output_chat])

    # Setup the flagging callbacks
    automatic_logging_text.setup(inputs_to_text_callback, flagging_dir='text_logs')
    automatic_logging_chat.setup(inputs_to_chat_callback, flagging_dir='chatbot_logs')

    # Change visibility of generation parameters if we perform greedy search
    do_sample.input(lambda value: [gr.update(visible=value) for _ in range(6)], inputs=do_sample,
                    outputs=[top_k, top_p, temperature, num_return_sequence, use_seed, seed])
    

if __name__ == '__main__':
    demo.queue().launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])
