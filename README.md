# anaconda-models

Download, launch, and integrate AI models curated by Anaconda.

Anaconda provides quantization files for a [curated collection](https://docs.anaconda.com/ai-navigator/user-guide/models/)
of large-language-models (LLMs).
This package provides programmatic access and an SDK to access the curated models, which are also available
to [Anaconda AI Navigator](https://docs.anaconda.com/ai-navigator/).

Below you will find documentation for

* [How to install](#install)
* [Command line interface to list, download, run API server for a model](#cli)
* [Integration with LLM CLI](#llm)
* [Anaconda Model Cache SDK](#sdk)
* [Langchain](#langchain)
* [LlamaIndex](#llamaindex)
* [LiteLLM](#litellm)
* [DSPy](#dspy)
* [PandasAI](#pandasai)
* [Panel ChatInterface](#panel)
* [Appendix: model download path](#download-path)

## Install

```text
conda install -c anaconda-cloud -c ai-staging anaconda-models
```

## How to authenticate

To use the Python client or CLI you can use `anaconda login` CLI, Anaconda Navigator, or

```python
from anaconda_cloud_auth import login
login()
```

to launch a browser to login and save your API token to disk.

For cases where you cannot utilize a browser to login, first use a desktop machine to print the API key

```text
anaconda cloud api-key
```

and on the non-interactive machine set the `ANACONDA_CLOUD_API_KEY=<api-key>` env var.

## Declaring model quantization files

In the CLI, SDK, and integrations below individual model quantizations are are referenced according the
following scheme.

```text
[<author>/]<model_name></ or _><quantization>[.<format>]
```

Fields surrounded by `[]` are optional.
The essential elements are the model name and quantization method
separated by either `/` or `_`. The supported quantization methods are

* Q4_K_M
* Q5_K_M
* Q6_K
* Q8_0

## CLI

```text
❯ anaconda models

 Usage: anaconda models [OPTIONS] COMMAND [ARGS]...

 Actions for Anaconda curated models

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────╮
│ download            Download a model                                                               │
│ info                Information about a single model                                               │
│ launch              Launch an inference server for a model                                         │
│ list                List models                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## listing

```text
❯ anaconda models list
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃                                             ┃             ┃            ┃ Quantizations     ┃                  ┃              ┃
┃                                             ┃             ┃            ┃ (downloaded in    ┃                  ┃              ┃
┃ Model                                       ┃ Type        ┃ Params (B) ┃ bold)             ┃ Trained for      ┃ License      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 01-ai/yi-34b-200k                           │ llama       │    34.39   │ Q4_K_M, Q5_K_M,   │ text-generation  │ other        │
│                                             │             │            │ Q6_K, Q8_0        │                  │              │
│ 01-ai/yi-6b                                 │ llama       │     6.06   │ Q4_K_M, Q5_K_M,   │ text-generation  │ other        │
│                                             │             │            │ Q6_K, Q8_0        │                  │              │
│ 01-ai/yi-6b-200k                            │ llama       │     6.06   │ Q4_K_M, Q5_K_M,   │ text-generation  │ other        │
│                                             │             │            │ Q6_K, Q8_0        │                  │              │
...
```

The `list` command accetps the option `--downloaded-only` to only list models for which one or more qunatization files have been downloaded.

```text
❯ anaconda models list --downloaded-only
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃                               ┃       ┃            ┃ Quantizations        ┃                     ┃         ┃
┃ Model                         ┃ Type  ┃ Params (B) ┃ (downloaded in bold) ┃ Trained for         ┃ License ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ BAAI/bge-small-en-v1.5        │ bert  │     0.03   │ Q4_K_M, Q5_K_M       │ sentence-similarity │ mit     │
│ meta-llama/llama-2-7b-chat-hf │ llama │     6.74   │ Q4_K_M               │ text-generation     │ llama2  │
└───────────────────────────────┴───────┴────────────┴──────────────────────┴─────────────────────┴─────────┘
```

## info

Detailed metadata is available for each model (including available quantizations)

```text
❯ anaconda models info TinyLlama/TinyLlama-1.1B-Chat-v1.0
                                                         TinyLlama/TinyLlama-1.1B-Chat-v1.0

   Description   The TinyLlama-1.1B-Chat-v1.0 model is a lightweight, 1.1 billion parameter chatbot designed for interactive conversational
                 applications. Developed as part of the TinyLlama project, this model uses the same architecture and tokenizer as Llama 2, ensuring
                 compatibility with various open-source projects. It was initially fine-tuned on the UltraChat dataset, which includes a range of
                 synthetic dialogues generated by ChatGPT, and further aligned using the DPOTrainer on the UltraFeedback dataset. This compact
                 model is optimized for efficient performance, making it suitable for environments with limited computational resources.

   Parameters    1.10 B

 Context Window  2048

   Trained For   text-generation

   Model Type    llama

   Source URL    https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

     License     apache-2.0

 First Published 2023-12-30T06:27:30+00:00

    Languages    en

 Quantized Files ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
                 ┃ Id                                             ┃ Method ┃ Format ┃ Evals                ┃ Max Ram (GB) ┃ Size (GB) ┃
                 ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
                 │ TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q8_0.gguf   │ Q8_0   │ GGUF   │ winogrande:  59.4059 │ 1.09         │ 1.09      │
                 │ TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf │ Q4_K_M │ GGUF   │ winogrande:  60.3960 │ 0.62         │ 0.62      │
                 │ TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q5_K_M.gguf │ Q5_K_M │ GGUF   │ winogrande:  61.3861 │ 0.73         │ 0.73      │
                 │ TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q6_K.gguf   │ Q6_K   │ GGUF   │ winogrande:  58.4158 │ 0.84         │ 0.84      │
                 └────────────────────────────────────────────────┴────────┴────────┴──────────────────────┴──────────────┴───────────┘

```

## download

To download a quantized model to local cache provide the full name of the file `[<author>/]<model>_<quantization>.gguf`. Use `--force` to download again. The author is optional.

The path to the downloaded file is printed.

```text
❯ anaconda models download TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf
Downloading TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 667.8/667.8 MB 72.1 MB/s 0:00:09
/Users/adefusco/Library/Caches/anaconda-models/TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf
```

## launch

To launch an inference service with llama.cpp the `launch` command takes several input values. The launch command uses the Intake 2 LlamaServerReader service to launch llaama.cpp. You must have llama.cpp installed.

```text
❯ anaconda models launch --help

 Usage: anaconda models launch [OPTIONS] MODEL_ID

 Launch an inference server for a model

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_id      TEXT  Name of the quantized model, it will download first if needed. [default: None] [required]         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --show    --no-show             Open your webbrowser when the inference service starts. [default: show]                    │
│ --host                 TEXT     IP address or hostname for the inference service. [default: 127.0.0.1]                     │
│ --port                 INTEGER  Port number for the inference service. [default: 8080]                                     │
│ --help                          Show this message and exit.                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## LLM

To use the llm integration you will need to also install `llm` package

```text
conda install -c conda-forge llm
```

then you can list models

```text
llm anaconda models
```

and chat with them, this will first ensure that the model has been downloaded and start llama.cpp in the background. OpenAI parameters are supported.

```text
llm -m 'anaconda:meta-llama/llama-2-7b-chat-hf_Q4_K_M.gguf' -o temperature 0.1 'what is pi?'
```

## SDK

The `AnacondaQuantizedModelCache` class provides the main functionality for

* metadata retrieval
* downloading the model
* starting llama.cpp server

```python
from anaconda_models import AnacondaQuantizeModelCache

model = AnacondaQuantizedModelCache(model="Llama-2-7B-Chat", quantization="q4_k_m")

# all the metadata about this quantization of the model
print(model.mdatadata)

# ensure that the model has been downloaded
path = model.download()

# start llama.cpp (this will download the model if not already done)
service = model.start()

# open the webbrowser for the llama.cpp server UI
service.open()

# Or get an OpenAI Client
client = service.openai_client()
```

### Server parameters

The `.start()` method accepts [llama.cpp server CLI flags](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#usage) as Python keyword-arguments, changing `-` to `_`. For example

```python
service = model.start(ctx_size=1024)
```

For CLI flags (without arguments), you can use the value `None`. For example to enable embeddings:

```python
service = model.start(embedding=None)
```

Unless `port=<integer>` is provide llama-server will start on a random unused port. You can
use the `service.url` attribute to determine the discovered port number.

### Server attributes

The output of `.start()` is a subclass of the Intake LlamaCPPService with a few extra attributes.

* `.openai_url`: is the url with `/v1` appended to utilize the OpenAI compatibility endpoints
* `.openai_client()`: creates a pre-configured OpenAI client for this url
* `.openai_async_client()`: creates a pre-configured Async OpenAI client for this url

Each of  `.openai_client()` and `opeanai_async_client()` allow extra keyword parameters to pass to the
client initialization.

## Langchain

The LangChain integration provides Chat and Embedding classes that automatically download and start the llama.cpp server

```python
from langchain.prompts import ChatPromptTemplate
from anaconda_models.langchain import AnacondaQuantizedModelChat AnacondaQuantizedModelEmbeddings

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = AnacondaQuantizedModelChat(model_name='meta-llama/llama-2-7b-chat-hf_Q4_K_M.gguf')

chain = prompt | model

message = chain.invoke({'topic': 'python'})
```

In addition to standard OpenAI parameters you can adjust llama.cpp server flags with `llama_cpp_options={...}`.

## LlamaIndex

LlamaIndex is support through a namespace package installed with `anaconda-models`. You will need at
least the `llama-index-llms-openai` package installed to use the integration.

```python
from llama_index.llms.anaconda_models import AnacondaModel

llm = AnacondaModel(
    model='OpenHermes-2.5-Mistral-7B_q4_k_m'
)
```

The `AnacondaModel` class supports the following arguments

* `model`: Name of the model using the pattern defined above
* `quantization`: Optional quantization method if not included in `model`
* `format`: Optional file format if not included in `model`
* `system_prompt`: Optional system prompt to apply to completions and chats
* `client`: Optional `anaconda_models.client.Client` object
* `llama_cpp_kwargs`: Optional dictionary of llama.cpp server parameters
* `temperature`: Optional temperature to apply to all completions and chats (default is 0.1)
* `max_tokens`: Optional Max tokens to predict (default is to let the model decide when to finish)

## LiteLLM

This provides a CustomLLM provider for use with `litellm`. But, since litellm does not currently support entrypoints to register the provider, the user must import the module first.

```python
import litellm
import anaconda_models.litellm

response = litellm.completion(
    'anaconda/openhermes-2.5-mistral-7b/q4_k_m',
    messages=[{'role': 'user', 'content': 'what is pi?'}]
)
```

Supported usage:

* completion (with and without stream=True)
* acompletion (with and without stream=True)
* Most OpenAI [inference parameters](https://docs.litellm.ai/docs/completion/input)
  * `n`: number of completions is not supported
* llama.cpp server options are passed as a dictionary called `llama_cpp_kwargs` (see above)

## DSPy

Since DSPy uses LiteLLM, Anaconda models can be used with dspy.
Streaming and async are supported for raw LLM calls and for modules
like Predict or ChainofThought
.

```python
import dspy
import anaconda_models.litellm

lm = dspy.LM('anaconda/openhermes-2.5-mistral-7b/q4_k_m')
dspy.configure(lm=lm)

chai = dspy.ChainOfThought("question -> answer")
chain(question="Who are you?")
```

## PandasAI

[PandasAI](https://github.com/Sinaptik-AI/pandas-ai): chat with data

```python
from anaconda_models.pandasai import AnacondaModel
from pandasai import SmartDataframe

llm = AnacondaModel(model_name='OpenHermes-2.5-Mistral-7B_q4_k_m.gguf', temperature=0.1)
sdf = SmartDataframe(df, config={'llm': llm})
sdf.chat('what is the average of this column where some condition is true?')
```

## Panel

A callback is available to work with Panel's [ChatInterface](https://panel.holoviz.org/reference/chat/ChatInterface.html)

To use it you will need to have panel, httpx, and openai installed.

```text
conda create -p ./pn -c anaconda-cloud/label/dev -c anaconda-cloud -c ai-staging anaconda-models panel httpx conda-forge::openai
```

Here's an example application that can be written in Python script or Jupyter Notebook

```python
import panel as pn
from anaconda_models.panel import AnacondaModelHandler
from anaconda_cloud_auth.client import BaseClient

pn.extension('echarts', 'tabulator', 'terminal')

aclient = BaseClient()

llm = AnacondaModelHandler('TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf', display_throughput=True)

chat = pn.chat.ChatInterface(
    callback=llm.callback,
    user=aclient.name,
    avatar=aclient.avatar,
    show_button_name=False)

chat.send(
    "I am your assistant. How can I help you?",
    user=llm.model_id, avatar=llm.avatar, respond=False
)
chat.servable()
```

## Download path

For all of the above integrations this package maintains a cache of the utilized model files at the AI Navigator download directory (even if you don't have AI Navigator installed).

```
~/.ai-navigator/models/
```

Within the directory model files are organized by model name.

```
❯ tree -R ~/.ai-navigator/models
/Users/adefusco/.ai-navigator/models
├── Qwen
│   └── Qwen1.5-7B-Chat
├── TinyLlama
│   └── TinyLlama-1.1B-Chat-v1.0
│       ├── TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf
│       └── TinyLlama-1.1B-Chat-v1.0_Q8_0.gguf
├── teknium
│   └── openhermes-2.5-mistral-7b
│       ├── OpenHermes-2.5-Mistral-7B_Q4_K_M.gguf
│       └── llama.log
└── tiiuae
    └── falcon-7b
        └── Falcon-7B_Q4_K_M.gguf
```

This directory can be changed using the `~/.anaconda/config.toml` file

```toml
[plugin.models]
cache_path = "<path>"
```

or using the `ANACONDA_MODELS_CACHE_PATH` environment variable.

## Setup for development

Ensure you have `conda` installed.
Then run:

```shell
make setup
```

## Run the unit tests

```shell
make test
```

## Run the unit tests across isolated environments with tox

```shell
make tox
```
