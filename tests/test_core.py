from itertools import product
import pytest

from anaconda_models.config import ModelsConfig
from anaconda_models.core import MODEL_NAME
from anaconda_models.core import AnacondaQuantizedModelCache
from anaconda_models.core import model_info
from anaconda_models.core import quantized_model_info


@pytest.mark.parametrize(
    "author,model,quantization,suffix",
    [
        ("TinyLlama", "TinyLlama-1.1B-Chat-v1.0", "Q4_K_M", "GGUF"),
        ("TinyLlama", "TinyLlama-1.1B-Chat-v1.0", "Q8_0", "GGUF"),
        ("tinyllama", "tinyllama-1.1b-chat-v1.0", "q4_k_m", "gguf"),
        ("tinyllama", "tinyllama-1.1b-chat-v1.0", "q8_0", "gguf"),
        ("Apple", "OpenELM-1_1B", "Q4_K_M", "GGUF"),
        ("apple", "openelm-1_1b", "q4_k_m", "gguf"),
        ("meta-llama", "Llama-2-7B-Chat", "Q4_K_M", "GGUF"),
        ("meta-llama", "llama-2-7b-chat", "q4_k_m", "gguf")
    ],
)
def test_model_name_regex(
    author: str, model: str, quantization: str, suffix: str
) -> None:
    match = MODEL_NAME.match(model)
    assert match
    assert match.groups() == (None, model, None, None)

    match = MODEL_NAME.match(f"{author}/{model}")
    assert match
    assert match.groups() == (author, model, None, None)

    match = MODEL_NAME.match(f"{author}/{model}_{quantization}")
    assert match
    assert match.groups() == (
        author,
        model,
        quantization,
        None,
    )
    match = MODEL_NAME.match(f"{author}/{model}_{quantization}.{suffix}")
    assert match
    assert match.groups() == (
        author,
        model,
        quantization,
        suffix,
    )

    match = MODEL_NAME.match(f"{author}/{model}/{quantization}")
    assert match
    assert match.groups() == (
        author,
        model,
        quantization,
        None,
    )
    match = MODEL_NAME.match(f"{author}/{model}/{quantization}.{suffix}")
    assert match
    assert match.groups() == (
        author,
        model,
        quantization,
        suffix,
    )

    match = MODEL_NAME.match(f"{model}_{quantization}")
    assert match
    assert match.groups() == (
        None,
        model,
        quantization,
        None,
    )
    match = MODEL_NAME.match(f"{model}_{quantization}.{suffix}")
    assert match
    assert match.groups() == (
        None,
        model,
        quantization,
        suffix,
    )

    match = MODEL_NAME.match(f"{model}/{quantization}")
    assert match
    assert match.groups() == (
        None,
        model,
        quantization,
        None,
    )
    match = MODEL_NAME.match(f"{model}/{quantization}.{suffix}")
    assert match
    assert match.groups() == (
        None,
        model,
        quantization,
        suffix,
    )


@pytest.mark.integration
def test_get_model_info() -> None:
    author = "TinyLlama"
    model = "TinyLlama-1.1B-Chat-v1.0"

    info = model_info(f"{author}/{model}")
    assert info["id"] == f"{author}/{model}"

    info = model_info(model)
    assert info["id"] == f"{author}/{model}"

    with pytest.raises(ValueError):
        _ = model_info("not-a-model/")

    with pytest.raises(ValueError):
        _ = model_info("author/not-a-model/q7_1000.xlsx")

    info = model_info("not/a-model")
    assert info is None


@pytest.mark.integration
def test_get_model_info_llama2() -> None:
    author = "meta-llama"
    model = "llama-2-7b-chat-hf"
    model_name = "Llama-2-7B-Chat"

    info = model_info(f"{author}/{model_name}")
    assert info["id"] == f"{author}/{model}"

    info = model_info(model_name)
    assert info["id"] == f"{author}/{model}"

    info = model_info(f"{author}/{model_name}".lower())
    assert info["id"] == f"{author}/{model}"

    info = model_info(model_name.lower())
    assert info["id"] == f"{author}/{model}"


@pytest.mark.integration
def test_get_model_info_llama2_with_hf_suffix() -> None:
    author = "meta-llama"
    model = "llama-2-7b-chat-hf"

    info = model_info(f"{author}/{model}")
    assert info["id"] == f"{author}/{model}"

    info = model_info(model)
    assert info["id"] == f"{author}/{model}"


@pytest.mark.integration
def test_quantized_model_info() -> None:
    model = "TinyLlama-1.1B-Chat-v1.0"
    quant = "Q4_K_M"
    format = "GGUF"

    with pytest.raises(ValueError):
        _ = quantized_model_info(model)

    with pytest.raises(ValueError):
        _ = quantized_model_info(f"{model}/{quant}", quantization=quant)

    info = quantized_model_info(model=model, quantization=quant)
    assert info["id"].endswith(f"{model}/{quant}")

    info = quantized_model_info(model=model, quantization=quant)
    assert info["id"].endswith(f"{model}/{quant}")

    info = quantized_model_info(model=f"{model}/{quant}")
    assert info["id"].endswith(f"{model}/{quant}")

    info = quantized_model_info(model=f"{model}/{quant}.{format}")
    assert info["id"].endswith(f"{model}/{quant}")

    info = quantized_model_info(model=f"{model}/{quant}", format=format)
    assert info["id"].endswith(f"{model}/{quant}")


@pytest.mark.integration
def test_quantized_model_info_with_underscore() -> None:
    model_id = "OpenELM-1_1B"
    model_name = "OpenELM-1_1B"
    quant = "Q4_K_M"

    with pytest.raises(ValueError):
        _ = quantized_model_info(model_name)

    with pytest.raises(ValueError):
        _ = quantized_model_info(f"{model_name}/{quant}", quantization=quant)

    info = quantized_model_info(model=model_name, quantization=quant)
    assert info["id"].endswith(f"{model_id}/{quant}")

    info = quantized_model_info(model=f"{model_name}/{quant}")
    assert info["id"].endswith(f"{model_id}/{quant}")



@pytest.mark.integration
def test_cache_file_name() -> None:
    author = "TinyLlama"
    model = "TinyLlama-1.1B-Chat-v1.0"
    quant = "Q4_K_M"
    format = "gguf"

    config = ModelsConfig()
    expected_cache_path = (
        config.cache_path / author / model / f"{model}_{quant}.{format}"
    )

    cacher = AnacondaQuantizedModelCache(f"{model}/{quant}.{format}", cache_path="")
    assert cacher._cache == expected_cache_path

    cacher = AnacondaQuantizedModelCache(f"{model}/{quant}", cache_path="")
    assert cacher._cache == expected_cache_path

    cacher = AnacondaQuantizedModelCache(
        f"{model}/{quant}", format=format, cache_path=""
    )
    assert cacher._cache == expected_cache_path


@pytest.mark.integration
def test_cache_file_name_llama2() -> None:
    author = "meta-llama"
    model = "llama-2-7b-chat-hf"
    model_name = "Llama-2-7B-Chat"
    quant = "Q4_K_M"
    format = "gguf"

    config = ModelsConfig()
    expected_cache_path = (
        config.cache_path / author / model / f"{model_name}_{quant}.{format}"
    )

    cacher = AnacondaQuantizedModelCache(f"{model_name}/{quant}", cache_path="")
    assert cacher._cache == expected_cache_path

    cacher = AnacondaQuantizedModelCache(f"{model_name}/{quant}".lower(), cache_path="")
    assert cacher._cache == expected_cache_path

    cacher = AnacondaQuantizedModelCache(
        f"{model_name}/{quant}", format=format, cache_path=""
    )
    assert cacher._cache == expected_cache_path

    cacher = AnacondaQuantizedModelCache(
        f"{model_name}/{quant}".lower(), format=format, cache_path=""
    )
    assert cacher._cache == expected_cache_path
