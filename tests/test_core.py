import pytest

from anaconda_ai.clients.base import MODEL_NAME


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
        ("meta-llama", "llama-2-7b-chat", "q4_k_m", "gguf"),
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
