import gc
import os
import argparse
import queue
import copy
from concurrent.futures import ThreadPoolExecutor

from transformers import TextIteratorStreamer
import torch
import gradio as gr

import engine
from engine import loader
from engine.streamer import TextContinuationStreamer
from engine.conversation_template import GenericConversation
from helpers import utils


# Default model to load at start-up
DEFAULT = 'llama2-7B-chat' if torch.cuda.is_available() else 'bloom-560M'

# Initialize global model (necessary not to reload the model for each new inference)
MODEL = engine.HFModel(DEFAULT)

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# This will be a mapping between users and current conversation, to reload them with page reload
CACHED_CONVERSATIONS = {}

# Need to define one logger per user
LOGGERS_TEXT = {}
LOGGERS_CHAT = {}


def update_model(conversation: GenericConversation, username: str, model_name: str, quantization_8bits: bool,
                 quantization_4bits: bool) -> tuple[GenericConversation, str, str, str, str, list[list]]:
    """Update the model in the global scope.

    Parameters
    ----------
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    username : str
        Username of current user if any.
    model_name : str
        The new model name.
    quantization_8bits : bool
        Whether to load in 8 bits,.
    quantization_4bits : bool
        Whether to load in 4 bits.

    Returns
    -------
    tuple[str, str, str, list[list]]
        Corresponds to components (conversation, conv_id, prompt_text, output_text, prompt_chat, output_chat).
    """
    
    global MODEL

    try:
        # If we ask for the same setup, do nothing
        if model_name == MODEL.model_name and quantization_8bits == MODEL.quantization_8bits and \
            quantization_4bits == MODEL.quantization_4bits:
            return conversation, conversation.id, '', '', '', [[None, None]]
    except NameError:
        pass

    if quantization_8bits and quantization_4bits:
        raise gr.Error('You cannot use both int8 and int4 quantization. Choose either one and try reloading.')
    
    if (quantization_8bits or quantization_4bits) and not torch.cuda.is_available():
        raise gr.Error('You cannot use quantization if you run without GPUs.')

    # Delete the variables if they exist (they should except if there was an error when loading a model at some point)
    # to save memory before loading the new one
    try:
        del MODEL
        gc.collect()
    except NameError:
        pass

    # Try loading the model
    try:
        MODEL = engine.HFModel(model_name, quantization_8bits=quantization_8bits,
                               quantization_4bits=quantization_4bits)
    except Exception as e:
        raise gr.Error(f'The following error happened during loading: {repr(e)}. Please retry or choose another one.')
    
    new_conv = MODEL.get_empty_conversation()
    if username != '':
        CACHED_CONVERSATIONS[username] = new_conv
    
    # Return values to clear the input and output textboxes, and input and output chatbot boxes
    return new_conv, new_conv.id, '', '', '', [[None, None]]


def text_generation(prompt: str, max_new_tokens: int, do_sample: bool, top_k: int, top_p: float,
                    temperature: float, use_seed: bool, seed: int) -> str:
    """Text generation.

    Parameters
    ----------
    prompt : str
        The prompt to the model.
    max_new_tokens : int
        How many new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int,
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).
    use_seed : bool
        Whether to use a fixed seed for reproducibility.
    seed : int
        An optional seed to force the generation to be reproducible.
    Returns
    -------
    str
        String containing the sequence generated.
    """
    
    if not use_seed:
        seed = None

    timeout = 20

    # To show text as it is being generated
    streamer = TextIteratorStreamer(MODEL.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(MODEL.generate_text, prompt, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                 top_k=top_k, top_p=top_p, temperature=temperature, seed=seed,
                                 truncate_prompt_from_output=True, streamer=streamer)
    
        # Get results from the streamer and yield it
        try:
            # Ask the streamer to skip prompt and reattach it here to avoid showing special prompt formatting
            generated_text = prompt
            for new_text in streamer:
                generated_text += new_text
                yield generated_text

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
        # Get actual result and yield it (which may be slightly different due to postprocessing)
        generated_text = future.result()
        yield prompt + generated_text


def chat_generation(conversation: GenericConversation, prompt: str, max_new_tokens: int, do_sample: bool,
                    top_k: int, top_p: float, temperature: float, use_seed: bool,
                    seed: int) -> tuple[str, GenericConversation, list[list]]:
    """Chat generation.

    Parameters
    ----------
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    prompt : str
        The prompt to the model.
    max_new_tokens : int
        How many new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).
    use_seed : bool
        Whether to use a fixed seed for reproducibility.
    seed : int
        An optional seed to force the generation to be reproducible.

    Returns
    -------
    tuple[str, GenericConversation, list[list]]
        Components (prompt_chat, conversation, output_chat)
    """
    
    if not use_seed:
        seed = None

    timeout = 20

    # To show text as it is being generated
    streamer = TextIteratorStreamer(MODEL.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    conv_copy.append_user_message(prompt)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(MODEL.generate_conversation, prompt, system_prompt='', conv_history=conversation,
                                 max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                 temperature=temperature, seed=seed, truncate_if_conv_too_long=True, streamer=streamer)
        
        # Get results from the streamer and yield it
        try:
            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield '', conv_copy, conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield '', conversation, conversation.to_gradio_format()



def continue_generation(conversation: GenericConversation, additional_max_new_tokens: int, do_sample: bool,
                        top_k: int, top_p: float, temperature: float, use_seed: bool,
                        seed: int) -> tuple[GenericConversation, list[list]]:
    """Continue the last turn of the model output.

    Parameters
    ----------
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    additional_max_new_tokens : int
        How many new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int,
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).
    use_seed : bool
        Whether to use a fixed seed for reproducibility.
    seed : int
        An optional seed to force the generation to be reproducible.

    Returns
    -------
    tuple[str, list[list]
        Components (conversation, output_chat).
    """

    if not use_seed:
        seed = None

    timeout = 20

    # To show text as it is being generated
    streamer = TextContinuationStreamer(MODEL.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(MODEL.continue_last_conversation_turn, conv_history=conversation,
                                 max_new_tokens=additional_max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                 temperature=temperature, seed=seed, truncate_if_conv_too_long=True, streamer=streamer)
        
        # Get results from the streamer and yield it
        try:
            generated_text = conv_copy.model_history_text[-1]
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield conv_copy, conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, conversation.to_gradio_format()



def retry_chat_generation(conversation: GenericConversation, max_new_tokens: int, do_sample: bool,
                          top_k: int, top_p: float, temperature: float, use_seed: bool,
                          seed: int) -> tuple[GenericConversation, list[list]]:
    """Chat generation.

    Parameters
    ----------
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    max_new_tokens : int
        How many new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).
    use_seed : bool
        Whether to use a fixed seed for reproducibility.
    seed : int
        An optional seed to force the generation to be reproducible.

    Returns
    -------
    tuple[GenericConversation, list[list]]
        Components (conversation, output_chat)
    """
    
    if not use_seed:
        seed = None

    timeout = 20

    # Remove last turn
    prompt = conversation.user_history_text[-1]
    _ = conversation.user_history_text.pop(-1)
    _ = conversation.model_history_text.pop(-1)

    # To show text as it is being generated
    streamer = TextIteratorStreamer(MODEL.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    conv_copy.append_user_message(prompt)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(MODEL.generate_conversation, prompt, system_prompt='', conv_history=conversation,
                                 max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                 temperature=temperature, seed=seed, truncate_if_conv_too_long=True, streamer=streamer)
        
        # Get results from the streamer and yield it
        try:
            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield conv_copy, conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, conversation.to_gradio_format()



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
        Return True if both the username and password match some credentials stored in `CREDENTIALS_FILE`. 
        False otherwise.
    """

    with open(CREDENTIALS_FILE, 'r') as file:
        # Read lines and remove whitespaces
        lines = [line.strip() for line in file.readlines() if line.strip() != '']

    valid_usernames = lines[0::2]
    valid_passwords = lines[1::2]

    if username in valid_usernames:
        index = valid_usernames.index(username)
        # Check that the password also matches at the corresponding index
        if password == valid_passwords[index]:
            return True
    
    return False
    

def clear_chatbot(username: str) -> tuple[GenericConversation, str, list[list]]:
    """Erase the conversation history and reinitialize the elements.

    Parameters
    ----------
    username : str
        The username of the current session if any.

    Returns
    -------
    tuple[GenericConversation, str, list[list]]
        Corresponds to the tuple of components (conversation, output_chat, conv_id)
    """

    # Create new conv object (we need a new unique id)
    conversation = MODEL.get_empty_conversation()
    if username != '':
        CACHED_CONVERSATIONS[username] = conversation

    return conversation, conversation.to_gradio_format(), conversation.id



def loading(request: gr.Request) -> tuple[GenericConversation, list[list], str, str, bool, bool]:
    """Retrieve username and all cached values at load time, and set the elements to the correct values.

    Parameters
    ----------
    request : gr.Request
        Request sent to the app.

    Returns
    -------
    tuple[GenericConversation, list[list], str, str, bool, bool, dict]
        Corresponds to the tuple of components (conversation, output_chat, conv_id, username, model_name,
        quantization_8bits, quantization_4bits, max_new_tokens)
    """

    # Retrieve username
    if request is not None:
        try:
            username = request.username
        except BaseException:
            username = ''
    
    if username is None:
        username = ''
    
    # Check if we have cached a value for the conversation to use
    if username != '':
        if username in CACHED_CONVERSATIONS.keys():
            actual_conv = CACHED_CONVERSATIONS[username]
        else:
            actual_conv = MODEL.get_empty_conversation()
            CACHED_CONVERSATIONS[username] = actual_conv
            LOGGERS_TEXT[username] = gr.CSVLogger()
            LOGGERS_CHAT[username] = gr.CSVLogger()

        LOGGERS_TEXT[username].setup(inputs_to_text_callback, flagging_dir=f'text_logs/{username}')
        LOGGERS_CHAT[username].setup(inputs_to_chat_callback, flagging_dir=f'chatbot_logs/{username}')

    else:
        actual_conv = MODEL.get_empty_conversation()

    conv_id = actual_conv.id

    model_name = MODEL.model_name
    int8 = MODEL.quantization_8bits
    int4 = MODEL.quantization_4bits
    
    return actual_conv, actual_conv.to_gradio_format(), conv_id, username, model_name, int8, int4, gr.update(maximum=MODEL.get_context_size())
    


# Define general elements of the UI (generation parameters)
model_name = gr.Dropdown(loader.ALLOWED_MODELS, value=DEFAULT, label='Model name',
                         info='Choose the model you want to use.', multiselect=False)
quantization_8bits = gr.Checkbox(value=False, label='int8 quantization', visible=torch.cuda.is_available())
quantization_4bits = gr.Checkbox(value=False, label='int4 quantization', visible=torch.cuda.is_available())
max_new_tokens = gr.Slider(32, 4096, value=512, step=32, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
max_additional_new_tokens = gr.Slider(16, 1028, value=128, step=16, label='Max additional new tokens',
                           info='New tokens to generate with "Continue last answer".')
do_sample = gr.Checkbox(value=True, label='Sampling', info=('Whether to incorporate randomness in generation. '
                                                            'If not selected, perform greedy search.'))
top_k = gr.Slider(0, 200, value=50, step=5, label='Top-k',
               info='How many tokens with max probability to consider. 0 to deactivate.')
top_p = gr.Slider(0, 1, value=0.90, step=0.01, label='Top-p',
              info='Probability density threshold for new tokens. 1 to deactivate.')
temperature = gr.Slider(0, 1, value=0.8, step=0.01, label='Temperature',
                        info='How to cool down the probability distribution.')
use_seed = gr.Checkbox(value=False, label='Use seed', info='Whether to use a fixed seed for reproducibility.')
seed = gr.Number(0, label='Seed', info='Seed for reproducibility.', precision=0)
load_button = gr.Button('Load model', variant='primary')

# Define elements of the simple generation Tab
prompt_text = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
output_text = gr.Textbox(label='Model output')
generate_button_text = gr.Button('▶️ Submit', variant='primary')
clear_button_text = gr.Button('🗑 Clear', variant='secondary')

# Define elements of the chatbot Tab
prompt_chat = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
output_chat = gr.Chatbot(label='Conversation')
generate_button_chat = gr.Button('▶️ Submit', variant='primary')
continue_button_chat = gr.Button('🔂 Continue', variant='primary')
retry_button_chat = gr.Button('🔄 Retry', variant='primary')
clear_button_chat = gr.Button('🗑 Clear')


conversation = gr.State(MODEL.get_empty_conversation())
# Define NON-VISIBLE elements: they are only used to keep track of variables and save them to the callback (States
# cannot be used in callbacks).
username = gr.Textbox('', label='Username', visible=False)
conv_id = gr.Textbox('', label='Conversation id', visible=False)

# Define the inputs for the main inference
inputs_to_simple_generation = [prompt_text, max_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]
inputs_to_chatbot = [conversation, prompt_chat, max_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]
inputs_to_chatbot_continuation = [conversation, max_additional_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]
inputs_to_chatbot_retry = [conversation, max_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]

# Define inputs for the logging callbacks
inputs_to_text_callback = [model_name, quantization_8bits, quantization_4bits, username,
                           *inputs_to_simple_generation, output_text]
inputs_to_chat_callback = [model_name, quantization_8bits, quantization_4bits, username, output_chat, conv_id,
                           max_new_tokens, max_additional_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]

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

    # state variables
    conversation.render()
    username.render()
    conv_id.render()

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
                    continue_button_chat.render()
                    retry_button_chat.render()
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
                # with gr.Row():
                do_sample.render()
                with gr.Row():
                    max_new_tokens.render()
                    max_additional_new_tokens.render()
                with gr.Row():
                    top_k.render()
                    top_p.render()
                with gr.Row():
                    temperature.render()
                with gr.Row():
                    use_seed.render()
                    seed.render()



    # Perform simple text generation when clicking the button
    generate_event1 = generate_button_text.click(text_generation, inputs=inputs_to_simple_generation,
                                                 outputs=output_text)
    # Add automatic callback on success
    generate_event1.success(lambda *args: LOGGERS_TEXT[args[3]].flag(args) if args[3] != '' else None,
                            inputs=inputs_to_text_callback, preprocess=False)

    # Perform chat generation when clicking the button
    generate_event2 = generate_button_chat.click(chat_generation, inputs=inputs_to_chatbot,
                                                 outputs=[prompt_chat, conversation, output_chat])

    # Add automatic callback on success
    generate_event2.success(lambda *args: LOGGERS_CHAT[args[3]].flag(args, flag_option='generation') if args[3] != '' \
                            else None, inputs=inputs_to_chat_callback, preprocess=False)
    
    # Continue generation when clicking the button
    generate_event3 = continue_button_chat.click(continue_generation, inputs=inputs_to_chatbot_continuation,
                                                 outputs=[conversation, output_chat])
    
    # Add automatic callback on success
    generate_event3.success(lambda *args: LOGGERS_CHAT[args[3]].flag(args, flag_option='continuation') if args[3] != '' \
                            else None, inputs=inputs_to_chat_callback, preprocess=False)
    
    # Continue generation when clicking the button
    generate_event4 = retry_button_chat.click(retry_chat_generation, inputs=inputs_to_chatbot_retry,
                                              outputs=[conversation, output_chat])
    
    # Add automatic callback on success
    generate_event4.success(lambda *args: LOGGERS_CHAT[args[3]].flag(args, flag_option='retry') if args[3] != '' \
                            else None, inputs=inputs_to_chat_callback, preprocess=False)

    # Switch the model loaded in memory when clicking
    load_button.click(update_model, inputs=[conversation, username, model_name, quantization_8bits, quantization_4bits],
                      outputs=[conversation, conv_id, prompt_text, output_text, prompt_chat, output_chat],
                      cancels=[generate_event1, generate_event2, generate_event3])
    
    # Clear the prompt and output boxes when clicking the button
    clear_button_text.click(lambda: ['', ''], outputs=[prompt_text, output_text], queue=False)
    clear_button_chat.click(clear_chatbot, inputs=[username], outputs=[conversation, output_chat, conv_id], queue=False)

    # Change visibility of generation parameters if we perform greedy search
    do_sample.input(lambda value: [gr.update(visible=value) for _ in range(5)], inputs=do_sample,
                    outputs=[top_k, top_p, temperature, use_seed, seed], queue=False)
    
    # Correctly display the model and quantization currently on memory if we refresh the page (instead of default
    # value for the elements) and correctly reset the chat output
    loading_events = demo.load(loading, outputs=[conversation, output_chat, conv_id, username, model_name,
                                                 quantization_8bits, quantization_4bits, max_new_tokens], queue=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Playground')
    parser.add_argument('--no_auth', action='store_true',
                        help='If given, will NOT require authentification to access the webapp.')
    
    args = parser.parse_args()
    no_auth = args.no_auth
    
    if no_auth:
        demo.queue(concurrency_count=4).launch(share=True, blocked_paths=[CREDENTIALS_FILE])
    else:
        demo.queue(concurrency_count=4).launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])
