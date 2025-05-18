import requests
from anaconda_ai.integrations.llama_index import (
    AnacondaModel,
    AnacondaEmbedding,
    AnacondaVectorStore,
)
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.core.settings import Settings

DOCS_URL = (
    "https://raw.githubusercontent.com/anaconda/anaconda-ai/refs/heads/main/README.md"
)

Settings.llm = AnacondaModel.load_from_spec("chat")
Settings.embed_model = AnacondaEmbedding.load_from_spec("embed")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

vector_store = AnacondaVectorStore.load_from_spec()


def get_data(url: str) -> Document:
    res = requests.get(url)
    res.raise_for_status()
    doc = Document(text=res.text)
    return doc


docs = get_data(DOCS_URL)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    [docs], storage_context=storage_context, show_progress=True
)
query_engine = index.as_query_engine()

response = query_engine.query(
    "Load a model from Anaconda using langchain and set the temperature to 0.1"
)
print(response.response)
