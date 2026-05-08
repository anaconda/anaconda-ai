import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, NamedTuple, Optional

import boto3
from botocore.exceptions import ClientError
from rich.console import Console
from rich.status import Status
from sagemaker.core.resources import Endpoint, EndpointConfig, Model as SageMakerModel
from sagemaker.core.shapes.shapes import (
    ContainerDefinition,
    ModelDataSource,
    ProductionVariant,
    ProductionVariantRoutingConfig,
    S3ModelDataSource,
    VpcConfig,
)

from anaconda_auth.config import AnacondaAuthConfig
from anaconda_auth.token import TokenInfo

from ..clients import AnacondaAIClient
from ..clients.base import QuantizedFile
from ..config import AnacondaAIConfig

logger = logging.getLogger(__name__)

# Region → ECR URI mapping. Populated as the image is published to more regions.
# Users can always override with image_uri= on AnacondaModel.
_IMAGE_URIS: Dict[str, str] = {
    # "us-east-1": "123456789.dkr.ecr.us-east-1.amazonaws.com/anaconda-sagemaker-runtime:latest",
    # "us-west-2": "123456789.dkr.ecr.us-west-2.amazonaws.com/anaconda-sagemaker-runtime:latest",
}

_CODEBUILD_PROJECT_NAME = "anaconda-model-stage"
_IAM_ROLE_NAME = "AnacondaModelStageRole"
_SSM_API_KEY_PARAM = "/anaconda/api-key"

_BUILDSPEC = """\
version: 0.2
env:
  parameter-store:
    ANACONDA_AUTH_API_KEY: /anaconda/api-key
  variables:
    MODEL_UUID: ""
    FILE_UUID: ""
    ANACONDA_DOMAIN: "anaconda.com"
    TARGET_BUCKET: ""
    TARGET_KEY: ""
    EXPECTED_SIZE: ""
    EXPECTED_SHA256: ""
phases:
  pre_build:
    commands:
      - |
        DOWNLOAD_URL=$(curl -sfS \\
          "https://${ANACONDA_DOMAIN}/api/ai/model/models/${MODEL_UUID}/files/${FILE_UUID}/download" \\
          -H "Authorization: Bearer ${ANACONDA_AUTH_API_KEY}" \\
          -H "X-Anaconda-Api-Version: 2" \\
          | python3 -c "import sys,json; print(json.load(sys.stdin)['download_url'])")
  build:
    commands:
      - echo "[STAGE] Downloading to s3://${TARGET_BUCKET}/${TARGET_KEY}"
      - |
        curl -fSL "$DOWNLOAD_URL" \\
          | tee >(sha256sum > /tmp/checksum.txt) \\
          | aws s3 cp - "s3://${TARGET_BUCKET}/${TARGET_KEY}" \\
              --expected-size "$EXPECTED_SIZE"
  post_build:
    commands:
      - ACTUAL_SHA=$(awk '{print $1}' /tmp/checksum.txt)
      - |
        if [ "$ACTUAL_SHA" != "$EXPECTED_SHA256" ]; then
          echo "[STAGE] CHECKSUM MISMATCH expected=$EXPECTED_SHA256 actual=$ACTUAL_SHA"
          aws s3 rm "s3://${TARGET_BUCKET}/${TARGET_KEY}"
          exit 1
        fi
      - echo "[STAGE] Checksum verified"
      - echo "[STAGE] COMPLETE"
"""

_CODEBUILD_ASSUME_ROLE_POLICY = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "codebuild.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
)


def _set_env_var(value: Optional[Any], key: str, env: Dict[str, str]) -> Dict[str, str]:
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


_DEFAULT_SAGEMAKER_ROLE_NAME = "AmazonSageMaker-DefaultRole"

_SAGEMAKER_TRUST_POLICY = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
)


def _ensure_default_sagemaker_role(boto_session: boto3.Session) -> str:
    iam = boto_session.client("iam")

    try:
        resp = iam.get_role(RoleName=_DEFAULT_SAGEMAKER_ROLE_NAME)
        return resp["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    logger.warning(
        "Creating default SageMaker execution role: %s", _DEFAULT_SAGEMAKER_ROLE_NAME
    )
    iam.create_role(
        RoleName=_DEFAULT_SAGEMAKER_ROLE_NAME,
        AssumeRolePolicyDocument=_SAGEMAKER_TRUST_POLICY,
    )
    iam.attach_role_policy(
        RoleName=_DEFAULT_SAGEMAKER_ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    )

    resp = iam.get_role(RoleName=_DEFAULT_SAGEMAKER_ROLE_NAME)
    return resp["Role"]["Arn"]


def _resolve_role(role: Optional[str], boto_session: boto3.Session) -> str:
    if role is not None:
        return role

    sts = boto_session.client("sts")
    arn = sts.get_caller_identity()["Arn"]

    # SSO and service-linked roles can't be SageMaker execution roles
    if "aws-reserved/" in arn or "AWSReservedSSO" in arn or "AWSServiceRole" in arn:
        return _ensure_default_sagemaker_role(boto_session)

    # assumed-role → role ARN (e.g. SageMaker notebook instance)
    if ":assumed-role/" in arn:
        parts = arn.split("/")
        account = arn.split(":")[4]
        role_name = parts[-2]
        if "aws-reserved" not in role_name:
            return f"arn:aws:iam::{account}:role/{role_name}"

    # Already a role (e.g. running in SageMaker Studio)
    if ":role/" in arn and "aws-reserved/" not in arn:
        return arn

    return _ensure_default_sagemaker_role(boto_session)


class _StageConfig(NamedTuple):
    bucket: str
    prefix: str
    region: Optional[str]


def _default_stage_bucket(boto_session: boto3.Session) -> str:
    sts = boto_session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    return f"sagemaker-anaconda-staging-{account_id}"


def _ensure_bucket_exists(
    bucket: str, boto_session: boto3.Session, region: Optional[str] = None
) -> None:
    s3 = boto_session.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            create_kwargs: Dict[str, Any] = {"Bucket": bucket}
            resolved_region = region or boto_session.region_name
            if resolved_region and resolved_region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": resolved_region
                }
            s3.create_bucket(**create_kwargs)
            logger.info("Created S3 bucket: %s", bucket)
        else:
            raise


def _resolve_stage_config(
    boto_session: boto3.Session,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    region: Optional[str] = None,
) -> _StageConfig:
    ai_config = AnacondaAIConfig()
    stage_cfg = ai_config.stage.sagemaker

    resolved_bucket = bucket or stage_cfg.bucket or _default_stage_bucket(boto_session)
    resolved_prefix = prefix or stage_cfg.prefix
    resolved_region = region or stage_cfg.region

    return _StageConfig(
        bucket=resolved_bucket,
        prefix=resolved_prefix,
        region=resolved_region,
    )


def _sanitize_name(model_id: str) -> str:
    """SageMaker resource names must match [a-zA-Z0-9]([\\-a-zA-Z0-9]*[a-zA-Z0-9])?"""
    name = model_id.lower()
    name = re.sub(r"[^a-z0-9-]", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-")


def _model_config_hash(
    image_uri: str,
    env: Dict[str, str],
    role: str,
    model_data_source: Optional[ModelDataSource],
) -> str:
    """Deterministic short hash of the model config for reuse."""
    data_source_str = ""
    if model_data_source and model_data_source.s3_data_source:
        ds = model_data_source.s3_data_source
        data_source_str = f"{ds.s3_uri}|{ds.s3_data_type}|{ds.compression_type}"

    content = json.dumps(
        {"image": image_uri, "env": env, "role": role, "data": data_source_str},
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _s3_key_for_model(prefix: str, model_id: str) -> str:
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    return f"{prefix}{model_id}/model.gguf"


class AnacondaModel:
    def __init__(
        self,
        model_id: str,
        # Auth / site
        anaconda_api_key: Optional[str] = None,
        site: Optional[str] = None,
        anaconda_domain: Optional[str] = None,
        # SageMaker
        role: Optional[str] = None,
        image_uri: Optional[str] = None,
        region: Optional[str] = None,
        aws_profile: Optional[str] = None,
        # llama.cpp tuning
        ctx_size: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        parallel: Optional[int] = None,
        flash_attn: Optional[bool] = None,
        cont_batching: Optional[bool] = None,
        batch_size: Optional[int] = None,
        ubatch_size: Optional[int] = None,
        threads: Optional[int] = None,
        threads_http: Optional[int] = None,
        cache_type_k: Optional[str] = None,
        cache_type_v: Optional[str] = None,
        mlock: Optional[bool] = None,
        jinja: Optional[bool] = None,
        reasoning: Optional[str] = None,
        reasoning_budget: Optional[int] = None,
        # Container tuning
        inference_timeout: Optional[int] = None,
        health_timeout: Optional[int] = None,
        log_request_body: Optional[bool] = None,
    ):
        self.model_id = model_id
        self.anaconda_api_key = anaconda_api_key
        self.site = site
        self.anaconda_domain = anaconda_domain
        self.region = region
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.parallel = parallel
        self.flash_attn = flash_attn
        self.cont_batching = cont_batching
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.threads = threads
        self.threads_http = threads_http
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.mlock = mlock
        self.jinja = jinja
        self.reasoning = reasoning
        self.reasoning_budget = reasoning_budget
        self.inference_timeout = inference_timeout
        self.health_timeout = health_timeout
        self.log_request_body = log_request_body

        self._boto_session = boto3.Session(profile_name=aws_profile, region_name=region)

        credentials = self._boto_session.get_credentials()
        if credentials is None:
            profile_hint = " --aws-profile <name>" if not aws_profile else ""
            raise ValueError(
                "No AWS credentials found. Either:\n"
                f"  - Pass aws_profile= (or CLI:{profile_hint})\n"
                "  - Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY env vars\n"
                "  - Configure ~/.aws/credentials\n"
                "  - Run: aws configure"
            )

        self._quantized_file: Optional[QuantizedFile] = None
        self._staged_s3_uri: Optional[str] = None
        self._built_model: Optional[SageMakerModel] = None

        self._validate_model_id()
        self.env = self._configure_environment_variables()

        self.role = role
        self.image_uri = image_uri

    @property
    def quantized_file(self) -> QuantizedFile:
        if self._quantized_file is None:
            raise RuntimeError("Model ID has not been validated yet")
        return self._quantized_file

    def _validate_model_id(self) -> None:
        client = AnacondaAIClient(
            site=self.site,
            domain=self.anaconda_domain,
            backend="ai-catalyst",
        )
        self._quantized_file = client.models._find_quantization(self.model_id)
        self.model_id = (
            f"{self._quantized_file._model.name}/{self._quantized_file.quant_method}"
        )

    def _configure_environment_variables(self) -> Dict[str, str]:
        env: Dict[str, str] = {}

        resolved_key = _resolve_api_key(
            self.anaconda_api_key, self.site, self.anaconda_domain
        )
        env = _set_env_var(self.model_id, "ANACONDA_MODEL_ID", env)
        env = _set_env_var(resolved_key, "ANACONDA_AUTH_API_KEY", env)
        env = _set_env_var(self.anaconda_domain, "ANACONDA_AUTH_DOMAIN", env)

        # llama.cpp tuning
        env = _set_env_var(self.ctx_size, "LLAMA_ARG_CTX_SIZE", env)
        env = _set_env_var(self.n_gpu_layers, "LLAMA_ARG_N_GPU_LAYERS", env)
        env = _set_env_var(self.parallel, "LLAMA_ARG_N_PARALLEL", env)
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
        env = _set_env_var(self.batch_size, "LLAMA_ARG_BATCH", env)
        env = _set_env_var(self.ubatch_size, "LLAMA_ARG_UBATCH", env)
        env = _set_env_var(self.threads, "LLAMA_ARG_THREADS", env)
        env = _set_env_var(self.threads_http, "LLAMA_ARG_THREADS_HTTP", env)
        env = _set_env_var(self.cache_type_k, "LLAMA_ARG_CACHE_TYPE_K", env)
        env = _set_env_var(self.cache_type_v, "LLAMA_ARG_CACHE_TYPE_V", env)
        env = _set_env_var(
            int(self.mlock) if self.mlock is not None else None,
            "LLAMA_ARG_MLOCK",
            env,
        )

        # llama.cpp chat
        env = _set_env_var(
            int(self.jinja) if self.jinja is not None else None,
            "LLAMA_ARG_JINJA",
            env,
        )
        env = _set_env_var(self.reasoning, "LLAMA_ARG_REASONING", env)
        env = _set_env_var(self.reasoning_budget, "LLAMA_ARG_THINK_BUDGET", env)

        # Container tuning
        env = _set_env_var(self.inference_timeout, "INFERENCE_TIMEOUT_SECONDS", env)
        env = _set_env_var(self.health_timeout, "HEALTH_TIMEOUT_SECONDS", env)
        env = _set_env_var(
            int(self.log_request_body) if self.log_request_body is not None else None,
            "LOG_REQUEST_BODY",
            env,
        )

        return env

    # ------------------------------------------------------------------
    # S3 staging
    # ------------------------------------------------------------------

    def _is_staged(self, bucket: str, key: str, region: Optional[str] = None) -> bool:
        s3 = self._boto_session.client("s3", region_name=region)
        try:
            resp = s3.head_object(Bucket=bucket, Key=key)
            remote_size = resp["ContentLength"]
            return remote_size == self.quantized_file.size_bytes
        except ClientError:
            return False

    def _ensure_staging_infra(
        self,
        region: str,
        bucket: str,
        api_key: str,
        console: Console,
    ) -> str:
        iam = self._boto_session.client("iam", region_name=region)
        codebuild = self._boto_session.client("codebuild", region_name=region)
        ssm = self._boto_session.client("ssm", region_name=region)
        sts = self._boto_session.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]

        # --- IAM Role ---
        role_arn = f"arn:aws:iam::{account_id}:role/{_IAM_ROLE_NAME}"
        try:
            iam.get_role(RoleName=_IAM_ROLE_NAME)
            logger.debug("IAM role %s exists", _IAM_ROLE_NAME)
        except iam.exceptions.NoSuchEntityException:
            console.print(f"  Creating IAM role ({_IAM_ROLE_NAME})...")
            iam.create_role(
                RoleName=_IAM_ROLE_NAME,
                AssumeRolePolicyDocument=_CODEBUILD_ASSUME_ROLE_POLICY,
                Description="Anaconda model staging via CodeBuild",
            )
            # IAM role propagation takes a few seconds
            time.sleep(10)
            console.print("  IAM role created ✓")

        stage_policy = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:PutObject",
                            "s3:GetObject",
                            "s3:DeleteObject",
                        ],
                        "Resource": f"arn:aws:s3:::{bucket}/*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                        ],
                        "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/codebuild/{_CODEBUILD_PROJECT_NAME}*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["ssm:GetParameters"],
                        "Resource": f"arn:aws:ssm:{region}:{account_id}:parameter{_SSM_API_KEY_PARAM}",
                    },
                ],
            }
        )
        iam.put_role_policy(
            RoleName=_IAM_ROLE_NAME,
            PolicyName="AnacondaModelStagePolicy",
            PolicyDocument=stage_policy,
        )

        # --- SSM Parameter ---
        try:
            ssm.get_parameter(Name=_SSM_API_KEY_PARAM, WithDecryption=False)
            logger.debug("SSM parameter %s exists", _SSM_API_KEY_PARAM)
        except ssm.exceptions.ParameterNotFound:
            console.print(f"  Storing API key in SSM ({_SSM_API_KEY_PARAM})...")
            ssm.put_parameter(
                Name=_SSM_API_KEY_PARAM,
                Value=api_key,
                Type="SecureString",
                Description="Anaconda API key for model staging",
            )
            console.print("  SSM parameter created ✓")

        # --- CodeBuild Project ---
        try:
            codebuild.batch_get_projects(names=[_CODEBUILD_PROJECT_NAME])["projects"][0]
            logger.debug("CodeBuild project %s exists", _CODEBUILD_PROJECT_NAME)
        except (IndexError, KeyError):
            console.print(
                f"  Creating CodeBuild project ({_CODEBUILD_PROJECT_NAME})..."
            )
            codebuild.create_project(
                name=_CODEBUILD_PROJECT_NAME,
                description="Anaconda model staging — downloads from catalog, uploads to S3",
                source={"type": "NO_SOURCE", "buildspec": _BUILDSPEC},
                artifacts={"type": "NO_ARTIFACTS"},
                environment={
                    "type": "LINUX_CONTAINER",
                    "computeType": "BUILD_GENERAL1_MEDIUM",
                    "image": "aws/codebuild/amazonlinux2-x86_64-standard:5.0",
                },
                serviceRole=role_arn,
                timeoutInMinutes=60,
            )
            console.print("  CodeBuild project created ✓")

        return role_arn

    def _start_codebuild(
        self, bucket: str, key: str, region: Optional[str] = None
    ) -> str:
        codebuild = self._boto_session.client("codebuild", region_name=region)

        domain = self.anaconda_domain or "anaconda.com"
        quant = self.quantized_file

        overrides = [
            {
                "name": "MODEL_UUID",
                "value": str(getattr(quant, "model_uuid")),
                "type": "PLAINTEXT",
            },
            {
                "name": "FILE_UUID",
                "value": str(getattr(quant, "file_uuid")),
                "type": "PLAINTEXT",
            },
            {"name": "ANACONDA_DOMAIN", "value": domain, "type": "PLAINTEXT"},
            {"name": "TARGET_BUCKET", "value": bucket, "type": "PLAINTEXT"},
            {"name": "TARGET_KEY", "value": key, "type": "PLAINTEXT"},
            {
                "name": "EXPECTED_SIZE",
                "value": str(quant.size_bytes),
                "type": "PLAINTEXT",
            },
            {"name": "EXPECTED_SHA256", "value": quant.sha256, "type": "PLAINTEXT"},
        ]

        resp = codebuild.start_build(
            projectName=_CODEBUILD_PROJECT_NAME,
            environmentVariablesOverride=overrides,
        )
        return resp["build"]["id"]

    def _poll_progress(
        self,
        build_id: str,
        region: Optional[str] = None,
        console: Optional[Console] = None,
    ) -> None:
        codebuild = self._boto_session.client("codebuild", region_name=region)

        _console = console or Console()
        size_gb = self.quantized_file.size_bytes / (1024**3)
        status = "IN_PROGRESS"

        with _console.status(
            f"  Staging {self.model_id} ({size_gb:.2f} GB)..."
        ) as spinner:
            while True:
                build_resp = codebuild.batch_get_builds(ids=[build_id])
                build = build_resp["builds"][0]
                status = build["buildStatus"]
                phase = build.get("currentPhase", "QUEUED")

                if status != "IN_PROGRESS":
                    break

                spinner.update(
                    f"  Staging {self.model_id} ({size_gb:.2f} GB) — {phase}"
                )
                time.sleep(5)

        if status != "SUCCEEDED":
            log_group = f"/aws/codebuild/{_CODEBUILD_PROJECT_NAME}"
            log_stream = build_id.split(":")[-1]
            raise RuntimeError(
                f"CodeBuild staging failed with status={status}. "
                f"Build ID: {build_id}. Check CloudWatch logs at "
                f"{log_group}/{log_stream}"
            )

    def _stage_model(
        self,
        bucket: str,
        prefix: str,
        region: Optional[str] = None,
        console: Optional[Console] = None,
        force: bool = False,
    ) -> str:
        _console = console or Console()
        resolved_region = region or self._boto_session.region_name
        key = _s3_key_for_model(prefix, self.model_id)

        if not force and self._is_staged(bucket, key, resolved_region):
            _console.print(f"Model already staged at s3://{bucket}/{key}")
            return key

        _console.print(f"Checking staging infrastructure ({resolved_region})...")
        _ensure_bucket_exists(bucket, self._boto_session, resolved_region)
        api_key = _resolve_api_key(
            self.anaconda_api_key, self.site, self.anaconda_domain
        )
        self._ensure_staging_infra(resolved_region, bucket, api_key, _console)

        size_gb = self.quantized_file.size_bytes / (1024**3)
        _console.print(f"\nStaging {self.model_id} ({size_gb:.2f} GB)")

        build_id = self._start_codebuild(bucket, key, resolved_region)
        self._poll_progress(build_id, resolved_region, _console)
        _console.print("  ✓ staged")

        return key

    def stage(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        region: Optional[str] = None,
        force: bool = False,
        console: Optional[Console] = None,
    ) -> str:
        """Explicitly stage the model to S3. Returns the S3 URI.

        Can be called independently of deploy() for pre-staging in CI/CD.
        """
        cfg = _resolve_stage_config(self._boto_session, bucket, prefix, region)
        key = self._stage_model(
            cfg.bucket, cfg.prefix, cfg.region, console, force=force
        )
        self._staged_s3_uri = f"s3://{cfg.bucket}/{key}"
        return self._staged_s3_uri

    @property
    def is_staged(self) -> bool:
        return self._staged_s3_uri is not None

    def build(
        self,
        stage: bool = False,
        stage_bucket: Optional[str] = None,
        stage_prefix: Optional[str] = None,
        model_name: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> SageMakerModel:
        """Register a deployable model in SageMaker.

        If stage=True (default), stages the model to S3 first.
        If stage=False, the container will download from the Anaconda catalog at startup.
        """
        if stage and not self.is_staged:
            self.stage(bucket=stage_bucket, prefix=stage_prefix)

        role = self.role or _resolve_role(None, self._boto_session)
        image_uri = self.image_uri or _resolve_image_uri(
            None, self._boto_session.region_name
        )

        env = dict(self.env)
        model_data_source: Optional[ModelDataSource] = None

        if self.is_staged:
            assert self._staged_s3_uri is not None
            s3_prefix = self._staged_s3_uri.rsplit("/", 1)[0] + "/"
            model_data_source = ModelDataSource(
                s3_data_source=S3ModelDataSource(
                    s3_uri=s3_prefix,
                    s3_data_type="S3Prefix",
                    compression_type="None",
                )
            )
            env.pop("ANACONDA_AUTH_API_KEY", None)
            env.pop("ANACONDA_MODEL_ID", None)

        config_hash = _model_config_hash(image_uri, env, role, model_data_source)
        resolved_name = (
            model_name or f"anaconda-{_sanitize_name(self.model_id)}-{config_hash}"
        )

        region = self._boto_session.region_name

        try:
            sm_model = SageMakerModel.get(
                resolved_name, session=self._boto_session, region=region
            )
            logger.info("Reusing existing SageMaker model: %s", resolved_name)
        except Exception:
            container = ContainerDefinition(
                image=image_uri,
                environment=env,
                model_data_source=model_data_source,
            )
            sm_model = SageMakerModel.create(
                model_name=resolved_name,
                execution_role_arn=role,
                primary_container=container,
                tags=tags,
                session=self._boto_session,
                region=region,
            )

        self._built_model = sm_model
        return sm_model

    def deploy(
        self,
        instance_type: str,
        initial_instance_count: int = 1,
        endpoint_name: Optional[str] = None,
        stage: bool = False,
        stage_bucket: Optional[str] = None,
        stage_prefix: Optional[str] = None,
        container_startup_health_check_timeout: Optional[int] = None,
        model_data_download_timeout: Optional[int] = None,
        volume_size_in_gb: Optional[int] = None,
        routing_strategy: str = "LEAST_OUTSTANDING_REQUESTS",
        vpc_config: Optional[VpcConfig] = None,
        kms_key_id: Optional[str] = None,
        wait: bool = True,
        tags: Optional[list] = None,
    ) -> Endpoint:
        """Deploy to a SageMaker endpoint.

        Calls build() automatically if not already called.
        """
        if self._built_model is None:
            self.build(
                stage=stage,
                stage_bucket=stage_bucket,
                stage_prefix=stage_prefix,
                tags=tags,
            )

        if container_startup_health_check_timeout is None:
            container_startup_health_check_timeout = 600 if self.is_staged else 3600

        if endpoint_name:
            base_name = endpoint_name
        else:
            suffix = uuid.uuid4().hex[:8]
            base_name = f"anaconda-{_sanitize_name(self.model_id)}-{suffix}"
        config_name = f"{base_name}-config"
        ep_name = base_name

        region = self._boto_session.region_name

        assert self._built_model is not None
        variant = ProductionVariant(
            variant_name="AllTraffic",
            model_name=self._built_model.model_name,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            container_startup_health_check_timeout_in_seconds=container_startup_health_check_timeout,
            model_data_download_timeout_in_seconds=model_data_download_timeout,
            volume_size_in_gb=volume_size_in_gb,
            routing_config=ProductionVariantRoutingConfig(
                routing_strategy=routing_strategy
            ),
        )

        EndpointConfig.create(
            endpoint_config_name=config_name,
            production_variants=[variant],
            kms_key_id=kms_key_id,
            vpc_config=vpc_config,
            tags=tags,
            session=self._boto_session,
            region=region,
        )

        endpoint = Endpoint.create(
            endpoint_name=ep_name,
            endpoint_config_name=config_name,
            tags=tags,
            session=self._boto_session,
            region=region,
        )

        if wait:
            self._wait_for_endpoint(endpoint)

        return endpoint

    def _wait_for_endpoint(self, endpoint: Endpoint, poll: int = 10) -> None:
        console = Console()
        start = time.time()
        text = f"{endpoint.endpoint_name} (Creating)"

        with Status(text, console=console) as display:
            while True:
                endpoint.refresh()
                status = endpoint.endpoint_status
                elapsed = int(time.time() - start)
                minutes, seconds = divmod(elapsed, 60)

                if status == "InService":
                    break

                if status == "Failed":
                    reason = getattr(endpoint, "failure_reason", "unknown")
                    raise RuntimeError(f"Endpoint failed: {reason}")

                text = (
                    f"{endpoint.endpoint_name} ({status}) ({minutes}m {seconds:02d}s)"
                )
                display.update(text)
                time.sleep(poll)

        elapsed = int(time.time() - start)
        minutes, seconds = divmod(elapsed, 60)
        console.print(
            f"[bold green]✓[/] {endpoint.endpoint_name} (InService) ({minutes}m {seconds:02d}s)",
            highlight=False,
        )
