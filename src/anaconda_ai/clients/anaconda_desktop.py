from typing import List, Optional
from urllib.parse import quote

from pydantic import Field

from ..config import AnacondaAIConfig
from .base import Model, QuantizedFile, BaseModels, GenericClient


class AnacondaDesktopQuantizedFile(QuantizedFile):
    sha256: str = Field(alias="id")
    size_bytes: int = Field(alias="sizeBytes")
    quant_method: str = Field(alias="quantization")
    max_ram_usage: int = Field(alias="maxRamUsage")
    _model: "AnacondaDesktopModel"

    @property
    def is_downloaded(self) -> bool:
        model_id = quote(self._model.id, safe="")
        res = self._model._client.get(f"/api/models/{model_id}/files/{self.sha256}")
        res.raise_for_status()
        return res.json()["data"]["isDownloaded"]


class AnacondaDesktopModel(Model):
    id: str
    num_parameters: int = Field(alias="numParameters")
    context_window_size: int = Field(alias="contextWindowSize")
    trained_for: str = Field(alias="trainedFor")
    quantized_files: List[AnacondaDesktopQuantizedFile] = Field(alias="files")


class AnacondaDesktopModels(BaseModels):
    def list(self) -> List[AnacondaDesktopModel]:
        res = self._client.get("api/models")
        res.raise_for_status()

        data = res.json().get("data", [])

        models = []
        for entry in data:
            revised = {"id": entry["id"], "name": entry["name"], **entry["metadata"]}
            model = AnacondaDesktopModel(**revised)
            model._client = self._client
            models.append(model)

        return models


class AnacondaDesktopClient(GenericClient):
    def __init__(self, site: Optional[str] = None) -> None:
        _ = site  # just ignore it
        self._ai_config = AnacondaAIConfig()
        domain = f"localhost:{self._ai_config.backends.anaconda_desktop.port}"
        super().__init__(
            domain=domain, api_key=self._ai_config.backends.anaconda_desktop.api_key
        )
        self._base_uri = f"http://{domain}"

        self.models = AnacondaDesktopModels(self)
