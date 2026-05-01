import json
import logging
import os
from typing import Any, Dict, Iterator, Optional, Union

import boto3
from sagemaker.deserializers import JSONDeserializer
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.session import Session

from anaconda_auth.config import AnacondaAuthConfig
from anaconda_auth.token import TokenInfo

from ..clients import AnacondaAIClient

logger = logging.getLogger(__name__)

# Region → ECR URI mapping. Populated as the image is published to more regions.
# Users can always override with image_uri= on AnacondaModel.
_IMAGE_URIS: Dict[str, str] = {
    # "us-east-1": "123456789.dkr.ecr.us-east-1.amazonaws.com/anaconda-sagemaker-runtime:latest",
    # "us-west-2": "123456789.dkr.ecr.us-west-2.amazonaws.com/anaconda-sagemaker-runtime:latest",
}


def _set_env_var(value: Optional[Any], key: str, env: Dict[str, str]) -> Dict[str, str]:
    """Set an env var only if value is non-None and key is not already in env."""
    if value is None:
        return env
    if key not in env:
        env[key] = str(value)
    return env


def _resolve_api_key(
    anaconda_api_key: Optional[str] = None,
    site: Optional[str] = None,
    domain: Optional[str] = None,
) -> str:
    """Resolve Anaconda API key: explicit param → env var → keyring via TokenInfo."""
    if anaconda_api_key:
        return anaconda_api_key

    env_key = os.environ.get("ANACONDA_AUTH_API_KEY")
    if env_key:
        return env_key

    config_kwargs: Dict[str, Any] = {}
    if site is not None:
        config_kwargs["site"] = site
    if domain is not None:
        config_kwargs["domain"] = domain

    config = AnacondaAuthConfig(**config_kwargs)
    token = TokenInfo.load(domain=config.domain)
    if token.api_key:
        return token.api_key

    raise ValueError(
        "Could not resolve Anaconda API key. Provide anaconda_api_key=, "
        "set ANACONDA_AUTH_API_KEY, or log in with `anaconda login`."
    )


def _resolve_image_uri(image_uri: Optional[str], region: str) -> str:
    """Resolve ECR image URI from region map or user override."""
    if image_uri is not None:
        return image_uri

    uri = _IMAGE_URIS.get(region)
    if uri is None:
        raise ValueError(
            f"No pre-built Anaconda SageMaker image available for region '{region}'. "
            f"Available regions: {list(_IMAGE_URIS.keys()) or '(none yet)'}. "
            f"Provide image_uri= to use a custom container image."
        )
    return uri


def _resolve_role(role: Optional[str], sagemaker_session: Session) -> str:
    """Resolve IAM role: explicit param → SageMaker execution role."""
    if role is not None:
        return role

    try:
        from sagemaker import get_execution_role

        return get_execution_role(sagemaker_session=sagemaker_session)
    except ValueError:
        raise ValueError(
            "Could not auto-resolve SageMaker execution role. "
            "Pass role= explicitly or run from a SageMaker notebook/Studio environment."
        )


class AnacondaPredictor(Predictor):
    """Predictor with JSON serialization for Anaconda SageMaker endpoints."""

    def __init__(
        self,
        endpoint_name: str,
        sagemaker_session: Optional[Session] = None,
        serializer: Any = JSONSerializer(),
        deserializer: Any = JSONDeserializer(),
        component_name: Optional[str] = None,
    ):
        super().__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
            component_name=component_name,
        )

    def predict_stream(
        self, data: Dict[str, Any], **kwargs: Any
    ) -> Iterator[Dict[str, Any]]:
        """Invoke the endpoint with streaming. Returns an iterator of parsed SSE events.

        The container returns SSE when ``"stream": true`` is set in the request body.
        This method uses SageMaker's ``invoke_endpoint_with_response_stream`` API
        and yields each parsed JSON payload from the SSE stream.
        """
        data = dict(data, stream=True)

        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=self.sagemaker_session.boto_region_name,
        )
        response = runtime.invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=self.serializer.serialize(data),
            **kwargs,
        )

        for event in response["Body"]:
            chunk = event.get("PayloadPart", {}).get("Bytes", b"")
            if not chunk:
                continue

            for line in chunk.decode("utf-8").splitlines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    return

                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse SSE payload: %s", payload)
                    continue


class AnacondaModel(Model):
    """Deploy Anaconda AI Catalog models to SageMaker endpoints."""

    def __init__(
        self,
        model_id: str,
        # Auth / site
        anaconda_api_key: Optional[str] = None,
        site: Optional[str] = None,
        anaconda_domain: Optional[str] = None,
        # llama.cpp tuning
        ctx_size: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        parallel: Optional[int] = None,
        flash_attn: Optional[bool] = None,
        cont_batching: Optional[bool] = None,
        # Container tuning
        inference_timeout: Optional[int] = None,
        health_timeout: Optional[int] = None,
        log_request_body: Optional[bool] = None,
        # SageMaker pass-through
        predictor_cls: Any = AnacondaPredictor,
        **kwargs: Any,
    ):
        super().__init__(predictor_cls=predictor_cls, **kwargs)

        self.model_id = model_id
        self.anaconda_api_key = anaconda_api_key
        self.site = site
        self.anaconda_domain = anaconda_domain
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.parallel = parallel
        self.flash_attn = flash_attn
        self.cont_batching = cont_batching
        self.inference_timeout = inference_timeout
        self.health_timeout = health_timeout
        self.log_request_body = log_request_body

        self.sagemaker_session = self.sagemaker_session or Session()

        self._validate_model_id()
        self._initialize_model()

    def _validate_model_id(self) -> None:
        """Validate model_id against the Anaconda AI Catalog."""
        client = AnacondaAIClient(
            site=self.site,
            domain=self.anaconda_domain,
            backend="ai-catalyst",
        )
        # Raises ValueError / ModelNotFound / QuantizedFileNotFound if invalid
        client.models._find_quantization(self.model_id)

    def _initialize_model(self) -> None:
        """Configure env vars and resolve image URI."""
        self.env = self._configure_environment_variables()

        if self.image_uri is None:
            region = self.sagemaker_session.boto_region_name
            self.image_uri = _resolve_image_uri(None, region)

        if self.role is None:
            self.role = _resolve_role(None, self.sagemaker_session)

    def _configure_environment_variables(self) -> Dict[str, str]:
        env: Dict[str, str] = self.env.copy() if self.env else {}

        # Required
        resolved_key = _resolve_api_key(
            self.anaconda_api_key, self.site, self.anaconda_domain
        )
        env = _set_env_var(self.model_id, "ANACONDA_MODEL_ID", env)
        env = _set_env_var(resolved_key, "ANACONDA_AUTH_API_KEY", env)

        # Optional auth
        env = _set_env_var(self.anaconda_domain, "ANACONDA_AUTH_DOMAIN", env)

        # llama.cpp tuning
        env = _set_env_var(self.ctx_size, "LLAMA_ARG_CTX_SIZE", env)
        env = _set_env_var(self.n_gpu_layers, "LLAMA_ARG_N_GPU_LAYERS", env)
        env = _set_env_var(self.parallel, "LLAMA_ARG_PARALLEL", env)
        env = _set_env_var(
            int(self.flash_attn) if self.flash_attn is not None else None,
            "LLAMA_ARG_FLASH_ATTN",
            env,
        )
        env = _set_env_var(
            int(self.cont_batching) if self.cont_batching is not None else None,
            "LLAMA_ARG_CONT_BATCHING",
            env,
        )

        # Container tuning
        env = _set_env_var(self.inference_timeout, "INFERENCE_TIMEOUT_SECONDS", env)
        env = _set_env_var(self.health_timeout, "HEALTH_TIMEOUT_SECONDS", env)
        env = _set_env_var(
            int(self.log_request_body) if self.log_request_body is not None else None,
            "LOG_REQUEST_BODY",
            env,
        )

        return env

    def deploy(
        self,
        *args: Any,
        container_startup_health_check_timeout: Optional[int] = 3600,
        **kwargs: Any,
    ) -> Union[AnacondaPredictor, Any]:
        """Deploy the model to a SageMaker endpoint.

        Defaults container_startup_health_check_timeout to 3600 seconds
        to allow time for model download inside the container.
        """
        return super().deploy(
            *args,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
            **kwargs,
        )

    def compile(self, **_: Any) -> None:
        raise NotImplementedError(
            "AnacondaModel does not support SageMaker Neo compilation"
        )

    def transformer(self, **_: Any) -> None:
        raise NotImplementedError("AnacondaModel does not support Batch Transform")
