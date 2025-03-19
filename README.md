# anaconda-models

Download, launch, and integrate AI models curated by Anaconda.

Anaconda provides quantization files for a [curated collection](https://docs.anaconda.com/ai-navigator/user-guide/models/)
of large-language-models (LLMs).
This package provides programmatic access and an SDK to access the curated models, download them, and start servers.

Below you will find documentation for

* [How to install](#install)
* [Command line interface to list, download, run API servers for models](#cli)
* [Anaconda AI SDK](#sdk)
* [Integration with LLM CLI](#llm)
* [Langchain](#langchain)
* [LlamaIndex](#llamaindex)
* [LiteLLM](#litellm)
* [DSPy](#dspy)
* [PandasAI](#pandasai)
* [Panel ChatInterface](#panel)

## Install

```text
conda install -c anaconda-cloud anaconda-models
```

## Backend

The backend for anaconda-ai is [Anaconda AI Navigator](https://www.anaconda.com/products/ai-navigator). This package
package utilizes the backend API to list and download models and manage running servers. All activities performed
by the CLI, SDK, and integrations here are visible within Anaconda AI Navigator.

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

To download a quantized model to local cache provide the full name of the file `[<author>/]<model>_<quantization>.gguf`. The author is optional.

```text
❯ anaconda models download TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf
Downloading TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 667.8/667.8 MB 72.1 MB/s 0:00:09
/Users/adefusco/Library/Caches/anaconda-models/TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf
```

## launch

To launch an inference service with the backend the `launch` command takes several input values.

```text
❯ anaconda models launch --help

 Usage: anaconda models launch [OPTIONS] MODEL_ID

 Launch an inference server for a model

╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model      TEXT  Name of the quantized model or catalog entry, it will download first if needed.     │
│                       [default: None]                                                                     │
│                       [required]                                                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --show              --no-show                       Open your webbrowser when the inference service       │
│                                                     starts.                                               │
│                                                     [default: no-show]                                    │
│ --port                                     INTEGER  Port number for the inference service. Default is to  │
│                                                     find a free open port                                 │
│                                                     [default: 0]                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## SDK

The SDK actions are initiated by creating a client connection to the backend.

```python
from anaconda_ai import get_default_client

client = get_default_client()
```

The client provides two top-level accessors `.models` and `.servers`.

### Models

The `.models` attribute provides actions to list available models and download specific quantization files.

|Method|Return|Description|
|-----|-----|------|
|`.list()`|`List[ModelSummary]`|List all available and downloaded models|
|`.get('<model-name>')`|`ModelSummary`|retrieve metadata about a model|
|`.download('<model>/<quantization>')`|None|Download a model quantization file|

The `ModelSummary` class holds metadata for each available model

|Attribute/Method|Return|Description|
|---------|-------|--------|
|`.id`|string|The id of the model in the format `<author>/<model-name>`|
|`.name`|string|The name of the model|
|`.metadata`|`ModelMetadata`|Metadata about the model and quantization files|

The `ModelMetadata` holds

|Attribute/Method|Return|Description|
|---------|-------|--------|
|`.numParameters`|int|Number of parameters for the model|
|`.contextWindowSize`|int|Length of the context window for the model|
|`.trainedFor`|str|Either `'sentence-similarity'` or `'text-generation'`|
|`.description`|str|Description of the model provided by the original author|
|`.files`|`List[ModelQuantization]`|List of available quantization files|
|`.get_quantization('<method>')`|`ModelQuantization`|Retrieve metadata for a single quantization file|

Each `ModelQuantization` object provides

|Attribute/Method|Return|Description|
|---------|-------|--------|
|`id`|str|The sha256 checksum of the model file|
|`modelFileName`|str|The file name as it will appear on disk|
|`method`|str|The quantization method|
|`sizeBytes`|int|Size of the model file in bytes|
|`maxRamUsage`|int|The total amount of ram needed to load the model in bytes|
|`isDownloaded`|bool|True if the model file has been downloaded|
|`localPath`|str|Will be non-null if the model file has been downloaded|

#### Downloading models

The `.models.download('quantized-file-name')` method accepts two types of
input

* String name of the model with quantization, see above.
* a `ModelQuantization` object

If the model file has already been downloaded this function returns
immediately. Otherwise a progress bar is shown showing the download
progress.

### Servers

The `.servers` accessor provides methods to list running servers,
start new servers, and stop servers.

|Method|Return|Description|
|-----|-----|------|
|`.list`|`List[Server]`|List all running servers|
|`.match`|Server|Find a running server that matches supplied configuration|
|`.create`|Server|Create a new server configuration with supplied model file and API parameters|
|`.start`|None|Start the API server|
|`.status`|str|Return the status for a server id|
|`.stop('<server-id>')`|None|Stop a running server|
|`.delete('<server-id>')`|None|Completely remove record of server configuration|

#### Creating servers

The `.create` method will create a new server configuration. If there is already a running server with the same
model file and API parameters the matched server configuration is returned rather than creating and starting a new
server.

The `.create` function has the following inputs

|Argument|Type|Description|
|---|---|---|
|model|str or ModelQuantization|The string name for the quantized model or a ModelQuantization object|
|api_params|APIParams or dict|Parameters for how the server is configured, like host and port|
|load_params|LoadParams or dict|Control how the model is loaded, like n_gpu_layers, batch_size, or to enable embeddings|
|infer_params|InferParams or dict|Control inference configuration like sampling parameters, number of threads, or default temperature|

The three server parameters Pydantic classes are shown here.
If the value `None` is used for any parameter the server
will utilize the backend default value.

```python
class APIParams(BaseModel, extra="forbid"):
    host: str = "127.0.0.1"
    port: int = 0            # 0 means find a random unused port
    api_key: str | None = None
    log_disable: bool | None = None
    mmproj: str | None = None
    timeout: int | None = None
    verbose: bool | None = None
    n_gpu_layers: int | None = None
    main_gpu: int | None = None
    metrics: bool | None = None


class LoadParams(BaseModel, extra="forbid"):
    batch_size: int | None = None
    cont_batching: bool | None = None
    ctx_size: int | None = None
    main_gpu: int | None = None
    memory_f32: bool | None = None
    mlock: bool | None = None
    n_gpu_layers: int | None = None
    rope_freq_base: int | None = None
    rope_freq_scale: int | None = None
    seed: int | None = None
    tensor_split: list[int] | None = None
    use_mmap: bool | None = None
    embedding: bool | None = None


class InferParams(BaseModel, extra="forbid"):
    threads: int | None = None
    n_predict: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_last: int | None = None
    repeat_penalty: float | None = None
    temp: float | None = None
    parallel: int | None = None
```

For example to create a server with the OpenHermes model with
default values

```python
from anaconda_ai import get_default_client

client = get_default_client()
server = client.servers.create(
  'OpenHermes-2.5-Mistral-7B/Q4_K_M',
)
```

By default creating a server configuration will

* download the model file if needed
* run the server API on a random unused port

The optional server parameters listed above can be passed as dictionaries
as well as avoiding automatic model downloads. For example

```python
server = client.servers.create(
  'OpenHermes-2.5-Mistral-7B/Q4_K_M',
  api_params={"main_gpu": 1, "port": 9999},
  load_params={"ctx_size": 512, "n_gpu_layers": 10},
  infer_params={"temp": 0.1},
  download_if_needed=False
)
```

#### Starting servers

When a server is created it is not automatically started.
A server can be started and stopped in a number of ways

From the server object

```python
server.start()
server.stop()
```

From the `.servers` accessor

```python
client.servers.start(server)
client.servers.stop(server)
```

Alternatively you can use `.create` as a context manager, which will
automatically stop the server on exit of the indented block.

```python
with client.servers.create('OpenHermes-2.5-Mistral-7B/Q4_K_M') as server:
    openai_client = server.openai_client()
    # make requests to the server
```

### Server attributes

* `.url`: is the full url to the running server
* `.openai_url`: is the url with `/v1` appended to utilize the OpenAI compatibility endpoints
* `.openai_client()`: creates a pre-configured OpenAI client for this url
* `.openai_async_client()`: creates a pre-configured Async OpenAI client for this url

Each of  `.openai_client()` and `opeanai_async_client()` allow extra keyword parameters to pass to the
client initialization.

## LLM

To use the llm integration you will need to also install `llm` package

```text
conda install -c conda-forge llm
```

then you can list downloaded model quantizations with

```text
llm anaconda models
```

When utilizing a model it will first ensure that the model has been downloaded and start the server though the backend.
Standard OpenAI parameters are supported.

```text
llm -m 'anaconda:meta-llama/llama-2-7b-chat-hf_Q4_K_M.gguf' -o temperature 0.1 'what is pi?'
```


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
from anaconda_models.llama_index import AnacondaModel

llm = AnacondaModel(
    model='OpenHermes-2.5-Mistral-7B_q4_k_m'
)
```

The `AnacondaModel` class supports the following arguments

* `model`: Name of the model using the pattern defined above
* `system_prompt`: Optional system prompt to apply to completions and chats
* `client`: Optional `anaconda_models.client.AINavigator` or `anaconda_models.client.KuratorClient` object
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
