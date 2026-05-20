from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from typer.testing import CliRunner

from anaconda_cli_base.cli import app
from anaconda_ai.clients.ai_catalyst import (
    AICatalystModels,
    AICatalystQuantizedFile,
    AICatalystCollection,
)
from anaconda_ai.clients.ai_navigator import AINavigatorModels
from anaconda_ai.exceptions import QuantizedFileNotFound


SAMPLE_MODEL_UUID = "7106c922-eb21-4c10-b7bd-6017a58cfa2d"
SAMPLE_FILE_UUID = "4ab4ce09-d3a9-4d67-9dce-70575d0b5d0b"

SAMPLE_MANIFEST = {
    "file_uuid": SAMPLE_FILE_UUID,
    "model_uuid": SAMPLE_MODEL_UUID,
    "filename": "Qwen3-4B-Thinking-2507",
    "collection_type": "original",
    "format": "safetensors",
    "file_count": 2,
    "total_size_bytes": 200,
    "files": [
        {
            "filename": "model-00001-of-00002.safetensors",
            "size_bytes": 100,
            "sha256": "aaa",
            "download_path": f"/models/{SAMPLE_MODEL_UUID}/collections/{SAMPLE_FILE_UUID}/download/model-00001-of-00002.safetensors",
        },
        {
            "filename": "config.json",
            "size_bytes": 100,
            "sha256": "bbb",
            "download_path": f"/models/{SAMPLE_MODEL_UUID}/collections/{SAMPLE_FILE_UUID}/download/config.json",
        },
    ],
}


def _make_collection() -> AICatalystCollection:
    coll = AICatalystCollection(
        file_uuid=UUID(SAMPLE_FILE_UUID),
        model_uuid=UUID(SAMPLE_MODEL_UUID),
        filename="original-safetensors-collection",
        format="safetensors",
        collection_type="original",
        file_count=2,
        total_size_bytes=200,
        size_bytes=200,
        published=True,
    )
    mock_model = MagicMock()
    mock_model.name = "Qwen3-4B-Thinking-2507"
    coll._model = mock_model
    return coll


def _make_regular_quant() -> AICatalystQuantizedFile:
    quant = AICatalystQuantizedFile(
        file_uuid=UUID("11111111-1111-1111-1111-111111111111"),
        model_uuid=UUID(SAMPLE_MODEL_UUID),
        generated_on=datetime(2026, 5, 19, 8, 45, 21, tzinfo=timezone.utc),
        quant_engine="llama.cpp",
        published=True,
        context_window_size=4096,
        sha256="abc123",
        size_bytes=4000000,
        quant_method="Q4_K_M",
        format="gguf",
        max_ram_usage=6000000,
    )
    mock_model = MagicMock()
    mock_model.name = "Qwen3-4B-Thinking-2507"
    quant._model = mock_model
    return quant


@pytest.fixture()
def mock_catalyst_client() -> MagicMock:
    client = MagicMock()
    client.config = MagicMock()
    client.config.domain = "test.anaconda.com"
    return client


@pytest.fixture()
def catalyst_models(mock_catalyst_client: MagicMock) -> AICatalystModels:
    return AICatalystModels(mock_catalyst_client)


class TestGetCollection:
    def test_finds_collection(self) -> None:
        from anaconda_ai.clients.base import Model

        collection = _make_collection()
        regular_quant = _make_regular_quant()

        mock_client = MagicMock()
        mock_client.ai_config = MagicMock()

        model = Model(
            client=mock_client,
            name="TestModel",
            description="test",
            num_parameters=1000,
            trained_for="text-generation",
            context_window_size=4096,
            quantized_files=[regular_quant],
            collections=[collection],
        )

        result = model.get_collection("safetensors")
        assert result is collection
        assert result.filename == "original-safetensors-collection"
        assert result.collection_type == "original"

    def test_no_collection_raises(self) -> None:
        from anaconda_ai.clients.base import Model

        mock_client = MagicMock()
        mock_client.ai_config = MagicMock()

        model = Model(
            client=mock_client,
            name="TestModel",
            description="test",
            num_parameters=1000,
            trained_for="text-generation",
            context_window_size=4096,
            quantized_files=[],
            collections=[],
        )

        with pytest.raises(QuantizedFileNotFound, match="No safetensors collection"):
            model.get_collection("safetensors")


class TestDownloadCollectionCatalyst:
    def test_downloads_files_in_parallel(
        self,
        catalyst_models: AICatalystModels,
        mock_catalyst_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        collection = _make_collection()
        mock_model = MagicMock()
        mock_model.name = "Qwen3-4B-Thinking-2507"
        mock_model.get_collection.return_value = collection

        manifest_response = MagicMock()
        manifest_response.json.return_value = SAMPLE_MANIFEST
        manifest_response.raise_for_status = MagicMock()

        file_url_response = MagicMock()
        file_url_response.json.return_value = {
            "download_url": "https://signed.example.com/file"
        }
        file_url_response.raise_for_status = MagicMock()

        mock_catalyst_client.get.side_effect = [
            manifest_response,
            file_url_response,
            file_url_response,
        ]

        mock_stream_response = MagicMock()
        mock_stream_response.iter_content.return_value = [b"x" * 100]
        mock_stream_response.raise_for_status = MagicMock()

        with patch.object(catalyst_models, "get", return_value=mock_model):
            with patch(
                "anaconda_ai.clients.ai_catalyst.requests.get",
                return_value=mock_stream_response,
            ):
                catalyst_models.download_collection(
                    "Qwen3-4B-Thinking-2507",
                    path=tmp_path,
                    show_progress=False,
                )

        assert (tmp_path / "model-00001-of-00002.safetensors").exists()
        assert (tmp_path / "config.json").exists()

    def test_skips_already_downloaded(
        self,
        catalyst_models: AICatalystModels,
        mock_catalyst_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        collection = _make_collection()
        mock_model = MagicMock()
        mock_model.name = "Qwen3-4B-Thinking-2507"
        mock_model.get_collection.return_value = collection

        manifest_response = MagicMock()
        manifest_response.json.return_value = SAMPLE_MANIFEST
        manifest_response.raise_for_status = MagicMock()

        mock_catalyst_client.get.side_effect = [manifest_response]

        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 100)
        (tmp_path / "config.json").write_bytes(b"y" * 100)

        with patch.object(catalyst_models, "get", return_value=mock_model):
            catalyst_models.download_collection(
                "Qwen3-4B-Thinking-2507",
                path=tmp_path,
                show_progress=False,
            )

        assert mock_catalyst_client.get.call_count == 1

    def test_size_mismatch_raises(
        self,
        catalyst_models: AICatalystModels,
        mock_catalyst_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        collection = _make_collection()
        mock_model = MagicMock()
        mock_model.name = "Qwen3-4B-Thinking-2507"
        mock_model.get_collection.return_value = collection

        manifest_response = MagicMock()
        manifest_response.json.return_value = SAMPLE_MANIFEST
        manifest_response.raise_for_status = MagicMock()

        file_url_response = MagicMock()
        file_url_response.json.return_value = {
            "download_url": "https://signed.example.com/file"
        }
        file_url_response.raise_for_status = MagicMock()

        mock_catalyst_client.get.side_effect = [
            manifest_response,
            file_url_response,
            file_url_response,
        ]

        wrong_size_data = b"x" * 77
        mock_stream_response = MagicMock()
        mock_stream_response.iter_content.return_value = [wrong_size_data]
        mock_stream_response.raise_for_status = MagicMock()

        with patch.object(catalyst_models, "get", return_value=mock_model):
            with patch(
                "anaconda_ai.clients.ai_catalyst.requests.get",
                return_value=mock_stream_response,
            ):
                with pytest.raises(RuntimeError, match="Size mismatch"):
                    catalyst_models.download_collection(
                        "Qwen3-4B-Thinking-2507",
                        path=tmp_path,
                        show_progress=False,
                    )

    def test_unpublished_raises(
        self, catalyst_models: AICatalystModels, mock_catalyst_client: MagicMock
    ) -> None:
        collection = _make_collection()
        collection.published = False

        mock_model = MagicMock()
        mock_model.name = "Qwen3-4B-Thinking-2507"
        mock_model.get_collection.return_value = collection

        with patch.object(catalyst_models, "get", return_value=mock_model):
            with pytest.raises(RuntimeError, match="not published"):
                catalyst_models.download_collection("Qwen3-4B-Thinking-2507")


class TestDownloadCollectionNavigator:
    def test_raises_not_implemented(self) -> None:
        mock_client = MagicMock()
        nav_models = AINavigatorModels(mock_client)

        with pytest.raises(NotImplementedError, match="ai-catalyst"):
            nav_models.download_collection("SomeModel")


class TestCLISafetensorsFlag:
    def test_help_shows_safetensors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["ai", "download", "--help"])
        assert result.exit_code == 0
        assert "--safetensors" in result.stdout

    def test_rejects_quant_with_safetensors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        with patch("anaconda_ai.cli.AnacondaAIClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.models.download_collection.side_effect = ValueError(
                "Model/Q4_K_M does not look like a model name."
            )
            mock_cls.return_value = mock_client
            result = runner.invoke(
                app, ["ai", "download", "--safetensors", "Model/Q4_K_M"]
            )

        assert result.exit_code == 1
