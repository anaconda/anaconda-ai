from unittest.mock import MagicMock, call, patch

import pytest

from anaconda_ai.clients.ai_catalyst import AICatalystModels


@pytest.fixture()
def mock_catalyst_client() -> MagicMock:
    client = MagicMock()
    client.config = MagicMock()
    client.config.domain = "test.anaconda.com"
    return client


@pytest.fixture()
def catalyst_models(mock_catalyst_client: MagicMock) -> AICatalystModels:
    return AICatalystModels(mock_catalyst_client)


def _page(data: list) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"result": {"data": data, "total": len(data)}}
    response.raise_for_status = MagicMock()
    return response


def test_list_uses_models_v3_endpoint(
    catalyst_models: AICatalystModels, mock_catalyst_client: MagicMock
) -> None:
    mock_catalyst_client.get.return_value = _page([])

    assert catalyst_models.list() == []
    mock_catalyst_client.get.assert_called_once_with(
        "/api/ai/model/models",
        params={"limit": 1000, "offset": 0},
        headers={"X-Anaconda-Api-Version": "3"},
    )


def test_list_paginates_until_exhausted(
    catalyst_models: AICatalystModels, mock_catalyst_client: MagicMock
) -> None:
    full_page = [{"name": f"model-{i}"} for i in range(1000)]
    mock_catalyst_client.get.side_effect = [_page(full_page), _page([])]

    with patch("anaconda_ai.clients.ai_catalyst.AICatalystModel") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()
        models = catalyst_models.list()

    assert len(models) == 1000
    assert mock_catalyst_client.get.call_count == 2
    assert mock_catalyst_client.get.call_args_list[1] == call(
        "/api/ai/model/models",
        params={"limit": 1000, "offset": 1000},
        headers={"X-Anaconda-Api-Version": "3"},
    )
