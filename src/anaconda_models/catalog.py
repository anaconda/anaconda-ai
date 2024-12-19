from copy import deepcopy
from typing import Any
from typing import Optional

from intake.readers.convert import BaseConverter
from intake.readers.datatypes import GGUF
from intake.readers.datatypes import BaseData
from intake.readers.entry import Catalog
from intake.readers.readers import BaseReader

from anaconda_models.client import Client
from anaconda_models.core import AnacondaQuantizedModelCache
from anaconda_models.core import get_models
from anaconda_models.core import model_info
from anaconda_models.core import quantized_model_info


class AnacondaModel(BaseData):
    """A model from Anaconda"""

    structure = {"nested", "text"}

    def __init__(
        self, name: str, metadata: Optional[dict] = None, **client_kwargs: Any
    ) -> None:
        self.name: str = name
        self.client_kwargs = client_kwargs
        if metadata is None:
            client = Client(**self.client_kwargs)
            self.metadata = model_info(name, client=client)
        else:
            self.metadata = metadata


class AnacondaQuantizedModel(BaseData):
    """A quantized model from Anaconda"""

    structure = {"nested", "text"}

    def __init__(
        self,
        name: str,
        metadata: Optional[dict] = None,
        client_options: Optional[dict] = None,
        llama_cpp_options: Optional[dict] = None,
    ) -> None:
        self.name: str = name
        self.client_options = {} if client_options is None else client_options
        self.llama_cpp_options = {} if llama_cpp_options is None else llama_cpp_options
        if metadata is None:
            client = Client(**self.client_options)
            self.metadata = quantized_model_info(name, client=client)
        else:
            self.metadata = metadata


class AnacondaQuantizedModelReader(BaseReader):
    """A quantized model cached to local disk"""

    output_instance = "anaconda_models.core:AnacondaQuantizedModelCache"
    implements = {AnacondaQuantizedModel}

    def _read(
        self, data: AnacondaQuantizedModel, **client_kwargs: Any
    ) -> AnacondaQuantizedModelCache:
        self.client_kwargs = {**data.client_options, **client_kwargs}
        client = Client(**self.client_kwargs)
        model = AnacondaQuantizedModelCache(data.name, client=client)
        _ = model.download()
        return model


class AnacondaQuantizedModelToGGUF(BaseConverter):
    instances = {
        "anaconda_models.core:AnacondaQuantizedModelCache": "intake.readers.datatypes:GGUF"
    }

    def run(self, x: AnacondaQuantizedModelCache, **_: Any) -> GGUF:
        path = x.download()
        return GGUF(str(path), metadata=x.metadata)


class AnacondaModelReader(BaseReader):
    output_instance = "intake.readers.entry:Catalog"

    def discover(self) -> None:
        self.read()

    def _read(self, data: AnacondaModel, **client_kwargs: Any) -> Catalog:
        entries = []
        for quant in data.metadata["quantizedFiles"]:
            model_file_id = f"{data.metadata['modelId']}_{quant['quantMethod']}.{quant['format'].lower()}"
            qdata = AnacondaQuantizedModel(model_file_id, **client_kwargs)
            reader = AnacondaQuantizedModelReader(data=qdata)
            entries.append(reader)

        metadata = deepcopy(data.metadata)
        _ = metadata.pop("quantizedFiles")
        cat = Catalog(entries=entries, metadata=metadata)
        cat.aliases = {q.data.name: e for q, e in zip(entries, cat.entries)}
        return cat


class AnacondaModels(BaseReader):
    output_instance = "intake.readers.entry:Catalog"

    def discover(self) -> None:
        self.read()

    def _read(self, *_: Any, **client_kwargs: Any) -> Catalog:
        client = Client(**client_kwargs)
        models = get_models(client=client)

        entries = []
        for model in models:
            data = AnacondaModel(model["id"], metadata=model)
            reader = AnacondaModelReader(data=data, **client_kwargs)
            entries.append(reader)
        cat = Catalog(entries=entries)
        cat.aliases = {m.data.name: e for m, e in zip(entries, cat.entries)}
        return cat
