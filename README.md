# anaconda-ai

Download, launch, and integrate AI models curated by Anaconda.
This package provides programmatic access and an SDK to access the curated models, download them, and start servers.

Below you will find documentation for

* [How to install](#install)
* [Backends and configuration](#backends)
* [Command line interface to list, download, run API servers for models](#cli)
* [Anaconda AI SDK](#sdk)
* [Integration with LLM CLI](#llm)
* [Langchain](#langchain)
* [LlamaIndex](#llamaindex)
* [LiteLLM](#litellm)
* [DSPy](#dspy)
* [Pydantic AI](#pydanticai)
* [Instructor](#instructor)
* [Panel ChatInterface](#panel)
* [AWS Sagemaker](#sagemaker)
* [MCP Server](#mcp)

## Install

```text
conda install anaconda-ai
```

## Backends

The anaconda-ai package is the CLI/SDK for a number of backends that provide API endpoint to list and download models and manage running servers.
All activities performed by the CLI, SDK, and integrations here are visible within the backend application or site.

The available backends are

|Backend name|Configuration value|Supports|Default|
|------------|-------------------|--------|-------|
|[Anaconda AI Navigator](https://www.anaconda.com/products/ai-navigator)|`"ai-navigator"`|Models,Servers,Server Parameters,VectorDB|DEFAULT|
|Anaconda AI Catalyst|`"ai-catalyst"`|Models,Servers,Multi-Site||

## Configuration

Anaconda AI supports configuration management in the `~/.anaconda/config.toml` file. The following parameters are supported under the table `[plugin.ai]` or by setting
`ANACONDA_AI_<parameter>=<value>` environment variables.

|Parameter|Environment variable|Description|Default value|
|---------|--------------------|-----------|-------------|
|`backend`|`ANACONDA_AI_BACKEND`|The backend API|`"ai-navigator"`|
|`stop_server_on_exit`|`ANACONDA_AI_STOP_SERVER_ON_EXIT`|For any server started during a Python interpreter session stop the server when the interpreter stops. Does not affect servers that were previously running|`true`|
|`server_operations_timeout`|`ANACONDA_AI_SERVER_OPERATIONS_TIMEOUT`|Timeout waiting for a server to start or stop|`30`|
|`show_blocked_models`|`ANACONDA_AI_SHOW_BLOCKED_MODELS`|Toggle display of blocked models if backend supports it|`false`|

## Configuration CLI

Use `anaconda ai config` command to apply changes to the `~/.anaconda/config.toml`. See `anaconda ai config --help`
for details.

## Site configuration

For backends that support multi-site configuration see [anaconda-auth multi-site configuration](https://anaconda.github.io/anaconda-auth/#multi-site-configuration) and the [`anaconda sites`](https://anaconda.github.io/anaconda-auth/#cli-commands) CLI command.

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
|download|Download a model file using model name and quantization, or download a safetensors collection with `--safetensors`|
|launch|Launch a server for a model file|
|servers|Show all running servers or detailed information about a single server|
|stop|Stop a running server by id|
|launch-vectordb|Starts a pg vector db (not supported by all backends)|

See the `--help` for each command for more details.

## SDK

The SDK actions are initiated by creating a client connection to the backend.

```python
from anaconda_ai import AnacondaAIClient

client = AnacondaAIClient()
```

`AnacondaAIClient` extends [`BaseClient`](https://anaconda.github.io/anaconda-auth/#api-requests) from `anaconda-auth` with the following additional keyword arguments:

|Argument|Type|Description|Default|
|---|---|---|---|
|`backend`|str|The backend to use, see [Backends](#backends)|`"ai-navigator"`|
|`stop_server_on_exit`|bool|Stop servers started in this session when the Python interpreter exits|`True`|
|`server_operations_timeout`|int|Timeout in seconds waiting for a server to start or stop|`30`|

All other keyword arguments (e.g. `site`, `domain`, `api_key`, `ssl_verify`) are passed through to [`BaseClient`](https://anaconda.github.io/anaconda-auth/#api-requests).


The client provides two top-level accessors `.models` and `.servers`.

### Models

The `.models` attribute provides actions to list available models and download specific quantization files.

|Method|Return|Description|
|-----|-----|------|
|`.list()`|`List[Model]`|List all available and downloaded models|
|`.get('<model-name>')`|`Model`|retrieve metadata about a model|
|`.download('<model>/<quantization>')`|None|Download a model quantization file|
|`.download_collection('<model>')`|None|Download a safetensors collection (ai-catalyst only). Accepts `format=` kwarg (default: `"safetensors"`)|
|`.delete('<model>/<quantization>')`|None|Delete a downloaded model quantization file|

The `Model` class holds metadata for each available model

|Attribute/Method|Return|Description|
|---------|-------|--------|
|`.name`|string|The name of the model|
|`.description`|str|Description of the model provided by the original author|
|`.num_parameters`|int|Number of parameters for the model|
|`.trained_for`|str|Either `'sentence-similarity'` or `'text-generation'`|
|`.context_window_size`|int|Length of the context window for the model|
|`.quantized_files`|`List[QuantizedFile]`|List of available GGUF quantization files|
|`.collections`|`List[Collection]`|List of available file collections (e.g. safetensors)|
|`.get_quantization('<method>')`|`QuantizedFile`|Retrieve metadata for a single quantization file|
|`.get_collection('<format>')`|`Collection`|Retrieve a collection by format (default: `"safetensors"`)|
|`.download('<method>')`|None|Direct call to download a quantization file|
|`.delete('<method>')`|None|Delete a downloaded quantization file|

Each `QuantizedFile` object provides

|Attribute/Method|Return|Description|
|---------|-------|--------|
|`.identifier`|str|The file name as it will appear on disk|
|`.sha256`|str|The sha256 checksum of the model file|
|`.quant_method`|str|The quantization method|
|`.size_bytes`|int|Size of the model file in bytes|
|`.max_ram_usage`|int|The total amount of ram needed to load the model in bytes|
|`.is_downloaded`|bool|True if the model file has been downloaded|
|`.local_path`|Optional[Path]|Will be non-null if the model file has been downloaded|
|`.download()`|None|Direct call to download the quantization file|
|`.delete()`|None|Delete the downloaded quantization file|

#### Downloading models

There are three methods to download a quantization file:

1. Calling `.download()` from a `QuantizedFile` object
    * For example: `client.models.get('<model>').get_quantization('<method>').download()`
1. Calling `.download('<method>')` from a `Model` object
    * For example: `client.models.get('<model>').download('<method>')`
1. `client.models.download('quantized-file-name')`
    * the `.models.download()` method accepts two types of input: string name of the model with quantization or a `QuantizedFile` object

If the model file has already been downloaded this function returns
immediately. Otherwise a progress bar is shown showing the download
progress.

#### Collections

Collections group multiple related files (safetensors weight shards, tokenizer configs, chat templates, etc.)
into a single logical entry. Downloading a collection gives you everything needed to use the model with
the `transformers` library.

Each `Collection` object provides (ai-catalyst backend):

|Attribute|Return|Description|
|---------|------|-----------|
|`.file_uuid`|UUID|Unique identifier for the collection|
|`.model_uuid`|UUID|UUID of the parent model|
|`.filename`|str|Collection name|
|`.format`|str|Format of the collection (e.g. `"safetensors"`)|
|`.collection_type`|str|Type of collection (e.g. `"original"`)|
|`.file_count`|int|Number of files in the collection|
|`.total_size_bytes`|int|Combined size of all files|
|`.published`|bool|Whether the collection is published|

##### Downloading collections

Download a safetensors collection via the SDK:

```python
from anaconda_ai import AnacondaAIClient

client = AnacondaAIClient(backend="ai-catalyst")
client.models.download_collection("Qwen3-4B-Thinking-2507")
```

Or specify an output directory:

```python
client.models.download_collection("Qwen3-4B-Thinking-2507", path="./my-model")
```

The `format` kwarg defaults to `"safetensors"` and is the only supported value currently:

```python
client.models.download_collection("Qwen3-4B-Thinking-2507", format="safetensors")
```

Via the CLI:

```bash
anaconda ai download --safetensors Qwen3-4B-Thinking-2507
anaconda ai download --safetensors Qwen3-4B-Thinking-2507 --output ./my-model
```

Files are downloaded in parallel (5 concurrent) to the current directory by default,
or to the path specified by `--output`. The directory is created if it does not exist.
Each file shows its own progress bar. File sizes are verified after download.

Collection downloads are only supported with the `ai-catalyst` backend.

### Servers

The `.servers` accessor provides methods to list running servers,
start new servers, and stop servers.

|Method|Return|Description|
|-----|-----|------|
|`.list`|`List[Server]`|List all running servers|
|`.get('<server-id>')`|`Server`|Lookup server object by identifier|
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
|model|str or QuantizedFile|The string name for the quantized model or a QuantizedFile object|
|extra_options|dict|Control server configuration supported by the backend|

By default creating a server configuration will

* download the model file if required by the backend
* run the server API

For example to create a server with the OpenHermes model with
default values

```python
from anaconda_ai import AnacondaAIClient

client = AnacondaAIClient()
server = client.servers.create(
  'OpenHermes-2.5-Mistral-7B/Q4_K_M',
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

#### Server attributes

* `.status`: Text status of the server
* `.is_running`: Boolean status, True if the server is in the 'running' state
* `.start()`: Start the server, optional can be used as a context manager to auto stop
* `.stop()`: Stop the server
* `.url`: is the full url to the running server
* `.openai_url`: OpenAI compatibility url
* `.openai_client()`: creates a pre-configured OpenAI client for this url
* `.async_openai_client()`: creates a pre-configured Async OpenAI client for this url

Each of  `.openai_client()` and `async_openai_client()` allow extra keyword parameters to pass to the
client initialization.

#### Server Configuration Options

Not all backends support `extra_options=` on server create.

The AI Navigator backend supports [llama-server options](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage)
passed as snake-case dictionary keys to `client.servers.create()` with the `extra_options` kwarg.
To enable flags set the value to `True`.

Here are some notes on specific server parameter behavior

|Dict key|Notes|
|--------|-----|
|`port`|Start server on specific port, 0 or missing means start on random port|
|`jinja`|Set to `True` to enable tool calling for models trained to do so|

For example:

```python
from anaconda_ai import AnacondaAIClient

client = AnacondaAIClient()
server = client.servers.create(
  'OpenHermes-2.5-Mistral-7B/Q4_K_M',
  extra_options={
    "ctx_size": 512,
    "jinja": True
  }
)
```

### Vector Db

Creates a postgres vector db and returns the connection information. VectorDB is not supported by all backends.

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

Additionally, server configuration parameters like `ctx_size` can be passed

```text
llm -m 'anaconda:meta-llama/llama-2-7b-chat-hf_Q4_K_M.gguf' -o temperature 0.1 -o ctx_size 512 'what is pi?'
```

To use an already running server, use `server/<server-name>` as the model identifier:

```text
llm -m 'anaconda:server/my-server' -o temperature 0.1 'what is pi?'
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

To use an already running server, pass `server/<server-name>` as the `model_name`:

```python
model = AnacondaQuantizedModelChat(model_name='server/my-server')
```

The following keyword arguments are supported:

* `extra_options`: Dict, see create servers above

## LlamaIndex

You will need at least the `llama-index-llms-openai` package installed to use the integration.

```python
from anaconda_ai.integrations.llama_index import AnacondaModel

llm = AnacondaModel(
    model='OpenHermes-2.5-Mistral-7B_q4_k_m'
)
```

The `AnacondaModel` class supports the following arguments

* `model`: Name of the model using the pattern defined above, or `server/<server-name>` to use a running server
* `system_prompt`: Optional system prompt to apply to completions and chats
* `temperature`: Optional temperature to apply to all completions and chats (default is 0.1)
* `max_tokens`: Optional Max tokens to predict (default is to let the model decide when to finish)
* `extra_options`: Optional dict, see server creation above

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

To use an already running server, use `anaconda/server/<server-name>` as the model:

```python
response = litellm.completion(
    'anaconda/server/my-server',
    messages=[{'role': 'user', 'content': 'what is pi?'}]
)
```

Supported usage:

* completion (with and without stream=True)
* acompletion (with and without stream=True)
* Most OpenAI [inference parameters](https://docs.litellm.ai/docs/completion/input)
  * `n`: number of completions is not supported
* Server parameters can be passed as dictionaries to the `optional_params` keyword argument in the key "server"
  * `optional_params={"server": {"ctx_size": 512}}`

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

A running server can also be used: `dspy.LM('anaconda/server/my-server')`

`dspy.LM` supports `optional_params=` keyword argument as explained in the previous section.

## PydanticAI

The [Pydantic AI](https://ai.pydantic.dev/) integration provides ChatModel and EmbeddingModel support.
Here's an example using a chat model in an agent.

```python
from anaconda_ai.integrations.pydantic_ai import (
    AnacondaChatModel,
    AnacondaChatModelSettings,
)
settings = AnacondaChatModelSettings(temperature=0.1, extra_options={"ctx_size": 1024})

model = AnacondaChatModel(
    "OpenHermes-2.5-Mistral-7B/q4_k_m",
    settings=settings,
)
```

To use an already running server, pass `server/<server-name>` as the model name:

```python
model = AnacondaChatModel("server/my-server")
```

And embedding

```python
embed = AnacondaEmbeddingModel(
    "bge-small-en-v1.5/q4_k_m"
)

result = await embed.embed("cat", input_type="document")
```

You can also use `AnacondaEmbeddingModel("server/<name>")`.

## Instructor

This integration monkeypatches the `instructor.from_provider()` method on import. This is needed until the provider
can be added to the upstream [Instructor](https://python.useinstructor.com/) package.

```python
import instructor
from pydantic import BaseModel
import anaconda_ai.integrations.instructor  # noqa: F401

client = instructor.from_provider(
    "anaconda/OpenHermes-2.5-Mistral-7B/Q4_K_M", extra_options={"ctx_size": 512}
)

class UserInfo(BaseModel):
    name: str
    age: int


user_info = await client.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)
```

To use an already running server, use `anaconda/server/<server-name>`:

```python
client = instructor.from_provider("anaconda/server/my-server")
```

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

The `model_id`, first argument to `AnacondaModelHandler` can be `<model>/<quant>` format like in the
above example or `server/<server-name>`.

the AnacondaModelHandler supports the following keyword arguments

* `display_throughput`: Show a speed dial next to the response. Default is False
* `system_message`: Default system message applied to all responses
* `backend`: Anaconda AI backend, None means use default
* `site`: Anaconda Platform site for backends that support multiple sites, None means use default
* `client_options`: Optional dict passed as kwargs to chat.completions.create
* `extra_options`: Optional dict passed to `AnacondaAIClient.servers.create()`

## SageMaker

Deploy Anaconda AI Catalog models to AWS SageMaker real-time endpoints.

```text
pip install 'anaconda-ai[sagemaker]'
```

Requires the [anaconda-sagemaker-runtime](https://github.com/anaconda/anaconda-sagemaker-runtime)
container image built and pushed to your ECR repository.

### Python SDK

```python
from anaconda_ai.integrations.sagemaker import AnacondaModel
import json

IMAGE_URI = "<account_id>.dkr.ecr.<region>.amazonaws.com/anaconda-sagemaker-runtime:latest"

model = AnacondaModel(
    model_id="Qwen2.5-7B-Instruct/Q4_K_M",
    image_uri=IMAGE_URI,
)

# Deploy (container downloads model from catalog at startup)
endpoint = model.deploy(instance_type="ml.g5.2xlarge")

# Invoke
response = endpoint.invoke(
    body=json.dumps({"messages": [{"role": "user", "content": "What is conda?"}], "max_tokens": 256}),
    content_type="application/json",
)
print(json.loads(response.body))

# Cleanup
endpoint.delete()
```

For faster cold starts (~4 min vs ~6 min), pre-stage the model to S3:

```python
endpoint = model.deploy(instance_type="ml.g5.2xlarge", stage=True)
```

The `stage()` and `build()` steps can also be called explicitly:

```python
model.stage()                                          # upload GGUF to S3 via CodeBuild
model.build()                                          # register SageMaker Model resource
endpoint = model.deploy(instance_type="ml.g5.2xlarge") # create endpoint
```

### CLI

```bash
IMAGE_URI="<account_id>.dkr.ecr.<region>.amazonaws.com/anaconda-sagemaker-runtime:latest"

# Deploy (container downloads from catalog)
anaconda ai deploy Qwen2.5-7B-Instruct/Q4_K_M --image-uri $IMAGE_URI

# Deploy with S3 staging (faster cold start)
anaconda ai deploy Qwen2.5-7B-Instruct/Q4_K_M --image-uri $IMAGE_URI --stage

# Register model only (no endpoint), can be combined with --stage
anaconda ai deploy Qwen2.5-7B-Instruct/Q4_K_M --image-uri $IMAGE_URI --build-only

# Stage a model to S3 without deploying
anaconda ai stage Qwen2.5-7B-Instruct/Q4_K_M

# List staged models
anaconda ai stage --list
```

### llama-server tuning

```python
model = AnacondaModel(
    model_id="Qwen2.5-7B-Instruct/Q4_K_M",
    image_uri=IMAGE_URI,
    ctx_size=16384,
    parallel=8,
    flash_attn=True,
    cache_type_k="q8_0",
)
```

```bash
anaconda ai deploy Qwen2.5-7B-Instruct/Q4_K_M --image-uri $IMAGE_URI \
    --ctx-size 16384 --parallel 8 --flash-attn --cache-type-k q8_0
```
## MCP

An [MCP](https://modelcontextprotocol.io/) server exposing Anaconda AI model and server management as tools for LLM agents.

```text
pip install 'anaconda-ai[mcp]'
```

Start via CLI:

```text
anaconda ai mcp --transport stdio
```

Or via Python module:

```text
python -m anaconda_ai.mcp_server --transport stdio
```

### Configuration

Add to your MCP client config (e.g. Claude Desktop, Cursor):

```json
{
  "mcpServers": {
    "anaconda-ai": {
      "command": "anaconda",
      "args": ["ai", "mcp", "--transport", "stdio"]
    }
  }
}
```

If `anaconda` is not on PATH (e.g. inside a conda environment), use the full path to `python`:

```json
{
  "mcpServers": {
    "anaconda-ai": {
      "command": "/path/to/conda/env/bin/python",
      "args": ["-m", "anaconda_ai.mcp_server", "--transport", "stdio"]
    }
  }
}
```

### Available Tools

|Tool|Description|
|----|-----------|
|`list_models`|List models with quantizations, download status, and running status|
|`download_model`|Start downloading a quantized model file; returns immediately once download is confirmed in progress|
|`list_servers`|List running inference servers with status and URLs|
|`start_server`|Create and start an inference server for a quantized model|
|`stop_server`|Stop a running server by ID|
|`remove_server`|Remove a server configuration (optionally stopping it first)|

All tools accept optional `backend` and `site` parameters to target a specific backend.

The `start_server` tool accepts an `extra_options` dict supporting
[llama-server options](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage)
passed as snake_case keys (e.g. `ctx_size`, `jinja`, `n_gpu_layers`, `parallel`).
To enable boolean flags set the value to `true`.

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
