# anaconda-ai

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
* [Panel ChatInterface](#panel)

## Install

```text
conda install -c anaconda-cloud anaconda-ai
```

## Backend

The backend for anaconda-ai is [Anaconda AI Navigator](https://www.anaconda.com/products/ai-navigator). This package
package utilizes the backend API to list and download models and manage running servers. All activities performed
by the CLI, SDK, and integrations here are visible within Anaconda AI Navigator.

## Configuration

Anaconda AI supports configuration management in the `~/.anaconda/config.toml` file. The following parameters are supported under the table `[plugin.ai]` or by setting
`ANACONDA_AI_<parameter>=<value>` environment variables.

|Parameter|Environment variable|Description|Default value|
|---------|--------------------|-----------|-------------|
|`stop_server_on_exit`|`ANACONDA_AI_STOP_SERVER_ON_EXIT`|For any server started during a Python interpreter session stop the server when the interpreter stops. Does not affect servers that were previously running|`true`|

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

The CLI subcommands within `anaconda ai` provide full access to list and
download model files, start and stop servers through the backend.

|Command|Description|
|-------|-----------|
|models|Show all models or detailed information about a single model with downloaded model files indicated in bold|
|download|Download a model file using model name and quantization|
|launch|Launch a server for a model file|
|servers|Show all running servers or detailed information about a single server|
|stop|Stop a running server by id|
|launch-vectordb|Starts a pg vector db|

See the `--help` for each command for more details.

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
|`.download()`|None|Direct call to download the quantization file|
|`id`|str|The sha256 checksum of the model file|
|`modelFileName`|str|The file name as it will appear on disk|
|`method`|str|The quantization method|
|`sizeBytes`|int|Size of the model file in bytes|
|`maxRamUsage`|int|The total amount of ram needed to load the model in bytes|
|`isDownloaded`|bool|True if the model file has been downloaded|
|`localPath`|str|Will be non-null if the model file has been downloaded|

#### Downloading models

There are two methods to download a quantization file:

1. Calling `.download()` from a `ModelQuantization` object
    * For example: `client.models.get('<model>').get_quantization('<method>').download()`
1. `client.models.download('quantized-file-name')`
    * the `.models.download()` method accepts two types of input: string name of the model with quantization or a `ModelQuantization` object

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
|`.start('<server-id>')`|None|Start the API server|
|`.status('<server-id>')`|str|Return the status for a server id|
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

### Vector Db

Creates a postgres vector db and returns the connection information.

```text
anaconda ai launch-vectordb
```

## LLM

To use the llm integration you will need to also install `llm` package

```text
conda install -c conda-forge llm
```

then you can list downloaded model quantizations

```text
llm models
```

or to show only the Anaconda AI models

```text
llm models list -q anaconda
```

When utilizing a model it will first ensure that the model has been downloaded and start the server though the backend.
Standard OpenAI parameters are supported.

```text
llm -m 'anaconda:meta-llama/llama-2-7b-chat-hf_Q4_K_M.gguf' -o temperature 0.1 'what is pi?'
```

Standard OpenAI and the above server options are available for Anaconda AI models, to see the parameter names run

```text
llm models list -q anaconda --options
```

## Langchain

The LangChain integration provides Chat and Embedding classes that automatically manage downloading and starting servers.
You will need the `langchain-openai` package.

```python
from langchain.prompts import ChatPromptTemplate
from anaconda_ai.integrations.langchain import AnacondaQuantizedModelChat, AnacondaQuantizedModelEmbeddings

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = AnacondaQuantizedModelChat(model_name='meta-llama/llama-2-7b-chat-hf_Q4_K_M.gguf')

chain = prompt | model

message = chain.invoke({'topic': 'python'})
```

The following keyword arguments are supported:

* `api_params`: Dict or APIParams class above
* `load_params`: Dict or LoadParams class above
* `infer_params`: Dict or InferParams class above (excluding AnacondaQuantizedEmbedding)

## LlamaIndex

You will need at least the `llama-index-llms-openai` package installed to use the integration.

```python
from anaconda_ai.integrations.llama_index import AnacondaModel

llm = AnacondaModel(
    model='OpenHermes-2.5-Mistral-7B_q4_k_m'
)
```

The `AnacondaModel` class supports the following arguments

* `model`: Name of the model using the pattern defined above
* `system_prompt`: Optional system prompt to apply to completions and chats
* `temperature`: Optional temperature to apply to all completions and chats (default is 0.1)
* `max_tokens`: Optional Max tokens to predict (default is to let the model decide when to finish)
* `api_params`: Optional dict or APIParams object
* `load_params`: Optional dict or LoadParams object
* `infer_params`: Optional dict or InferParams object

## LiteLLM

This provides a CustomLLM provider for use with `litellm`.  But, since litellm does not currently support
[entrypoints](https://github.com/BerriAI/litellm/issues/7733) to register the provider,
the user must import the module first.

```python
import litellm
import anaconda_ai.integrations.litellm

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
* Server parameters (api_params, load_params, infer_params) can be passed as dictionaries to the `optional_params` keyword argument
  * `optional_params={"load_params": {"ctx_size": 512}}`

## DSPy

Since DSPy uses LiteLLM, Anaconda models can be used with dspy.
Streaming and async are supported for raw LLM calls and for modules
like Predict or ChainofThought
.

```python
import dspy
import anaconda_ai.integrations.litellm

lm = dspy.LM('anaconda/openhermes-2.5-mistral-7b/q4_k_m')
dspy.configure(lm=lm)

chain = dspy.ChainOfThought("question -> answer")
chain(question="Who are you?")
```

`dspy.LM` supports `optional_params=` keyword argument as explained in the previous section.

## Panel

A callback is available to work with Panel's [ChatInterface](https://panel.holoviz.org/reference/chat/ChatInterface.html)

To use it you will need to have panel, httpx, and numpy installed.

Here's an example application that can be written in Python script or Jupyter Notebook

```python
import panel as pn
from anaconda_ai.integrations.panel import AnacondaModelHandler

pn.extension('echarts', 'tabulator', 'terminal')

llm = AnacondaModelHandler('TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf', display_throughput=True)

chat = pn.chat.ChatInterface(
    callback=llm.callback,
    show_button_name=False)

chat.send(
    "I am your assistant. How can I help you?",
    user=llm.model_id, avatar=llm.avatar, respond=False
)
chat.servable()
```

the AnacondaModelHandler supports the following keyword arguments

* `display_throughput`: Show a speed dial next to the response. Default is False
* `system_message`: Default system message applied to all responses
* `client_options`: Optional dict passed as kwargs to chat.completions.create
* `api_params`: Optional dict or APIParams object
* `load_params`: Optional dict or LoadParams object
* `infer_params`: Optional dict or InferParams object

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
