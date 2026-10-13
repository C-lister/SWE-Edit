"""Helpers for rolling out the SWE-Edit agent on the continuous SWE-bench dataset.

This mirrors ``evaluation/OpenHands/utils.py`` — same container build / git diff /
continuous-dataset machinery — but the in-container agent is SWE-Edit
(https://github.com/microsoft/SWE-Edit), shipped as a wheel and driven by
``run_agent.py``.

Key differences vs. the OpenHands harness:
  * ``build_swebench_container`` copies the ``sweedit`` wheel + ``run_agent.py``
    into ``/agent`` instead of installing the OpenHands CLI.
  * the LLM credentials are passed in SWE-Edit's own env scheme
    (``API_TYPE`` / ``OPENAI_API_KEY`` / ``MODEL`` for the main agent and
    ``LLM_EDITOR_*`` for the viewer/editor subagents) rather than the
    ``LLM_MODEL`` / ``LLM_API_KEY`` LiteLLM scheme OpenHands wants.
"""

import io
import json
import logging
import os
import shlex
import tarfile
import tempfile
import threading
import traceback
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

import docker
import docker.errors
from swebench.harness.docker_build import close_logger, setup_logger
from swebench.harness.docker_utils import cleanup_container


# ---------------------------------------------------------------------------
# API configuration loading (.env)
# ---------------------------------------------------------------------------


def _parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() in ("true", "1", "yes")


def _load_config_for_api_type(api_type: str, prefix: str = "", suffix: str = "") -> dict[str, Any]:
    """Load one API configuration from environment variables."""
    config: dict[str, Any] = {"API_TYPE": api_type}

    if api_type == "OPENAI":
        if key := os.getenv(f"{prefix}OPENAI_API_KEY{suffix}"):
            config["OPENAI_API_KEY"] = key
        if url := os.getenv(f"{prefix}OPENAI_BASE_URL{suffix}"):
            config["OPENAI_BASE_URL"] = url
        if model := os.getenv(f"{prefix}MODEL{suffix}"):
            config["MODEL"] = model

    elif api_type == "AZURE_OPENAI":
        if key := os.getenv(f"{prefix}AZURE_API_KEY{suffix}"):
            config["AZURE_API_KEY"] = key
        if endpoint := os.getenv(f"{prefix}AZURE_ENDPOINT{suffix}"):
            config["AZURE_ENDPOINT"] = endpoint
        if deployment := os.getenv(f"{prefix}DEPLOYMENT{suffix}"):
            config["DEPLOYMENT"] = deployment
        if version := os.getenv(f"{prefix}AZURE_API_VERSION{suffix}"):
            config["AZURE_API_VERSION"] = version

    elif api_type == "ANTHROPIC":
        if key := os.getenv(f"{prefix}ANTHROPIC_API_KEY{suffix}"):
            config["ANTHROPIC_API_KEY"] = key
        if url := os.getenv(f"{prefix}ANTHROPIC_BASE_URL{suffix}"):
            config["ANTHROPIC_BASE_URL"] = url
        if model := os.getenv(f"{prefix}MODEL{suffix}"):
            config["MODEL"] = model

    elif api_type == "CLOUDGPT":
        if model := os.getenv(f"{prefix}MODEL{suffix}"):
            config["MODEL"] = model
        if (use_azure_cli := _parse_env_bool(os.getenv(f"{prefix}USE_AZURE_CLI{suffix}"))) is not None:
            config["USE_AZURE_CLI"] = use_azure_cli
        if (use_device_code := _parse_env_bool(os.getenv(f"{prefix}USE_DEVICE_CODE{suffix}"))) is not None:
            config["USE_DEVICE_CODE"] = use_device_code

    else:
        raise ValueError(f"Unknown API type: {api_type}")

    return config


def load_api_configs() -> list[dict[str, Any]]:
    """Load numbered or single API configurations from `.env`.

    Supports `API_TYPE_1`, `MODEL_1`, ... and falls back to `API_TYPE`, `MODEL`.
    """
    configs = []

    i = 1
    while True:
        api_type = os.getenv(f"API_TYPE_{i}")
        if not api_type:
            break
        configs.append(_load_config_for_api_type(api_type, suffix=f"_{i}"))
        i += 1

    if not configs and (api_type := os.getenv("API_TYPE")):
        configs.append(_load_config_for_api_type(api_type))

    return configs


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_global_logger(log_dir: str = "logs") -> logging.Logger:
    """Set up a global logger for the continuous SWE-Edit evaluation."""
    from datetime import datetime

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sweedit_continuous_{timestamp}.log")

    logger = logging.getLogger("sweedit_continuous")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_continuous_dataset(path: str) -> list[dict]:
    """Load and validate the continuous SWE-bench JSONL dataset.

    Each line is one sequence with ``continuous_id`` / ``repo`` / ``base_commit``
    / ``docker_image`` / ``bug_fixes``.
    """
    with open(path) as f:
        try:
            data = json.load(f)
            instances = data.get("instances", [])
        except json.JSONDecodeError:
            f.seek(0)
            instances = []
            for line in f:
                if line.strip():
                    instances.append(json.loads(line))

    if not instances:
        raise ValueError(f"No instances found in {path}")

    for seq in instances:
        assert "continuous_id" in seq, f"Missing continuous_id: {seq}"
        assert "repo" in seq, f"Missing repo: {seq.get('continuous_id')}"
        assert "base_commit" in seq, f"Missing base_commit: {seq.get('continuous_id')}"
        assert "bug_fixes" in seq, f"Missing bug_fixes: {seq.get('continuous_id')}"
        assert len(seq["bug_fixes"]) >= 2, f"Need >= 2 bug_fixes: {seq.get('continuous_id')}"

    return instances


# ---------------------------------------------------------------------------
# Git / patch helpers (work inside the workspace copy)
# ---------------------------------------------------------------------------


def apply_ground_truth_patch(container, workspace: str, patch_text: str, patch_id: str, logger=None) -> bool:
    """Apply a ground-truth patch via ``git apply`` (strict → reverse-check → 3way).

    Mirrors the OpenHands interleaved-mode replay so SWE-Edit sees the exact same
    prior-bug state. On success the patch is committed so subsequent diff
    baselines hold.
    """
    if not patch_text:
        if logger:
            logger.warning(f"No patch text provided for {patch_id}")
        return False

    patch_path = f"/{patch_id}.diff"
    container.exec_run(["bash", "-c", f"cat << 'EOF' > {patch_path}\n{patch_text}\nEOF"], user="root")

    apply_res = container.exec_run(
        ["bash", "-c", f"cd {workspace} && git apply -v {patch_path}"], user="root"
    )
    if apply_res.exit_code != 0:
        rev_res = container.exec_run(
            ["bash", "-c", f"cd {workspace} && git apply --reverse --check {patch_path}"],
            user="root",
        )
        if rev_res.exit_code == 0:
            if logger:
                logger.info(f"  ground-truth already applied for {patch_id}")
        else:
            threeway_res = container.exec_run(
                ["bash", "-c", f"cd {workspace} && git apply --3way -v {patch_path}"],
                user="root",
            )
            if threeway_res.exit_code == 0:
                if logger:
                    logger.info(f"  ground-truth applied via 3-way for {patch_id}")
            else:
                container.exec_run(
                    ["bash", "-c", f"cd {workspace} && git reset --hard HEAD && git clean -fd"],
                    user="root",
                )
                if logger:
                    err3 = threeway_res.output.decode("utf-8", errors="replace")
                    err1 = apply_res.output.decode("utf-8", errors="replace")
                    combined = (err3 or err1 or "").strip() or "<no stderr>"
                    logger.warning(
                        f"  ground-truth CONFLICT for {patch_id} (strict + 3way failed): {combined[:300]}"
                    )
                return False

    commit_cmd = (
        f"cd {workspace} && "
        f"git config user.email 'eval@sweedit.dev' && "
        f"git config user.name 'SWE-Edit' && "
        f"git add -A && "
        f"git commit -m 'ground-truth: {patch_id}' --allow-empty"
    )
    container.exec_run(["bash", "-c", commit_cmd], user="root")
    return True


def commit_agent_changes(container, workspace: str, message: str = "agent patch", logger=None) -> str:
    """Stage and commit everything the agent changed; return the new HEAD hash."""
    cmds = [
        f"cd {workspace} && git config user.email 'eval@sweedit.dev'",
        f"cd {workspace} && git config user.name 'SWE-Edit'",
        f"cd {workspace} && rm -f uv.lock",
        f"cd {workspace} && git checkout HEAD -- pyproject.toml tox.ini setup.py 2>/dev/null || true",
        f"cd {workspace} && git add -A",
        f"cd {workspace} && git commit -m '{message}' --allow-empty",
    ]
    for cmd in cmds:
        container.exec_run(["bash", "-c", cmd], user="root")

    result = container.exec_run(["bash", "-c", f"cd {workspace} && git rev-parse HEAD"], user="root")
    new_head = result.output.decode().strip()
    if logger:
        logger.info(f"Committed agent changes: {new_head[:12]}")
    return new_head


def get_git_diff_from_commit(container, workspace: str, from_commit: str, logger=None) -> str:
    """Get the diff between ``from_commit`` and current HEAD, excluding env files
    and binary blobs."""
    setup_files = ["pyproject.toml", "tox.ini", "setup.py"]
    exclude_patterns = " ".join([f"':!{f}'" for f in setup_files + ["uv.lock"]])

    diff_cmd = f"cd {workspace} && git --no-pager diff --no-color {from_commit} HEAD -- {exclude_patterns}"
    exec_result = container.exec_run(["bash", "-c", diff_cmd], user="root")

    if exec_result.exit_code != 0:
        if logger:
            logger.warning(f"Git diff failed: {exec_result.output.decode()}")
        return ""

    try:
        decoded = exec_result.output.decode("utf-8")
    except UnicodeDecodeError:
        output = exec_result.output
        clean_sections = []
        for i, section in enumerate(output.split(b"diff --git")):
            if i == 0 and not section.strip():
                continue
            try:
                decoded_section = section.decode("utf-8")
                if "Binary files" in decoded_section and "differ" in decoded_section:
                    continue
                clean_sections.append("diff --git" + decoded_section)
            except UnicodeDecodeError:
                continue
        return "".join(clean_sections)

    clean_sections = []
    for i, section in enumerate(decoded.split("diff --git")):
        if i == 0 and not section.strip():
            continue
        if "Binary files" in section and "differ" in section:
            continue
        clean_sections.append("diff --git" + section)

    return "".join(clean_sections)


# ---------------------------------------------------------------------------
# Container exec / file transfer
# ---------------------------------------------------------------------------


def exec_run_with_timeout(container, cmd, timeout=1200, logger=None, **kwargs):
    """Run a command in a container with a hard timeout, killing the agent on overrun."""
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = container.exec_run(cmd, **kwargs)
        except Exception as e:  # noqa: BLE001
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        try:
            kill_result = container.exec_run(["bash", "-c", "pkill -9 -f 'python|uv' || true"], user="root")
            if logger:
                logger.info(f"Killed container processes after timeout (exit_code={kill_result.exit_code})")
        except Exception as e:  # noqa: BLE001
            if logger:
                logger.warning(f"Failed to kill container processes: {e}")
        raise TimeoutError(f"Command execution timed out after {timeout} seconds")

    if exception[0]:
        raise exception[0]

    return result[0]


def copy_to_container_safe(container, src: Path, dst: Path):
    """Thread-safe copy of a local file into the container via a temp tar."""
    if os.path.dirname(dst) == "":
        raise ValueError(f"Destination path parent directory cannot be empty!, dst: {dst}")

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_tar:
        tar_path = Path(tmp_tar.name)

    try:
        with tarfile.open(tar_path, "w") as tar:
            tar.add(src, arcname=dst.name)
        with open(tar_path, "rb") as tar_file:
            data = tar_file.read()
        container.exec_run(f"mkdir -p {dst.parent}")
        container.put_archive(os.path.dirname(dst), data)
    finally:
        if tar_path.exists():
            tar_path.unlink()


def write_text_to_container(container, content: str, dst: Path, mode: int = 0o600):
    """Write UTF-8 text into a container without shell quoting."""
    data = content.encode("utf-8")
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        info = tarfile.TarInfo(name=dst.name)
        info.size = len(data)
        info.mode = mode
        tar.addfile(info, io.BytesIO(data))
    tar_stream.seek(0)

    container.exec_run(["mkdir", "-p", str(dst.parent)], user="root")
    container.put_archive(str(dst.parent), tar_stream.read())


# ---------------------------------------------------------------------------
# Container build
# ---------------------------------------------------------------------------


def build_swebench_container(
    client: docker.DockerClient,
    docker_image: str,
    whl_file_path: str,
    agent_script_path: str,
    run_id: str = "test_run",
    instance_id: str = None,
    env_vars: dict = None,
    mount: dict = None,
):
    """Pull ``docker_image``, start a container, install uv, and stage the
    SWE-Edit wheel + ``run_agent.py`` under ``/agent``.

    The continuous dataset's ``docker_image`` is the sole source of truth — we
    don't go through ``swebench.harness.test_spec`` / Hugging Face lookups.
    """
    assert docker_image, "docker_image is required"
    assert whl_file_path, "whl_file_path is required"
    assert agent_script_path, "agent_script_path is required"

    whl_file_path = Path(whl_file_path)
    agent_script_path = Path(agent_script_path)

    log_id = instance_id or run_id
    log_dir = Path("logs") / run_id / log_id
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_id, log_dir / "agent.log")

    try:
        client.images.get(docker_image)
    except docker.errors.ImageNotFound:
        logger.info(f"Pulling {docker_image}...")
        client.images.pull(docker_image)

    container_name = f"sweedit-{run_id[:40]}-{uuid.uuid4().hex[:6]}"
    with suppress(docker.errors.NotFound):
        client.containers.get(container_name).remove(force=True)

    create_kwargs = dict(
        image=docker_image,
        name=container_name,
        user="root",
        detach=True,
        entrypoint="",  # some swebench-pro images set ENTRYPOINT=/bin/bash
        command=["tail", "-f", "/dev/null"],
    )
    if mount:
        create_kwargs["volumes"] = {
            mount["local_path"]: {"bind": mount["container_path"], "mode": "rw"}
        }

    container = client.containers.create(**create_kwargs)
    container.start()
    logger.info(f"Container started: {container_name} ({docker_image})")

    # 1. Install uv system-wide and symlink to /usr/local/bin so plain `uv` works.
    logger.info("Installing uv...")
    install_uv_cmd = (
        "set -e; "
        "curl -LsSf https://astral.sh/uv/install.sh | sh; "
        "UV_PATH=$(find / -maxdepth 6 -name uv -type f -executable 2>/dev/null | head -1); "
        '[ -n "$UV_PATH" ] || { echo "uv binary not found after install" >&2; exit 1; }; '
        'ln -sf "$UV_PATH" /usr/local/bin/uv'
    )
    exec_result = container.exec_run(["bash", "-c", install_uv_cmd], user="root")
    if exec_result.exit_code != 0:
        raise Exception(f"Failed to install uv: {exec_result.output.decode()}")

    # 2. Stage the wheel + run_agent.py under /agent.
    logger.info("Copying sweedit wheel...")
    copy_to_container_safe(container, whl_file_path, Path(f"/agent/{whl_file_path.name}"))
    logger.info("Copying run_agent.py...")
    copy_to_container_safe(container, agent_script_path, Path("/agent/run_agent.py"))

    # 3. Pre-warm the wheel's dependency cache so the first bug's run doesn't
    #    eat the install cost inside its timeout budget.
    logger.info("Pre-warming sweedit dependency cache...")
    prewarm = container.exec_run(
        [
            "bash",
            "-c",
            f"uv run --python 3.12 --with /agent/{whl_file_path.name} python -c 'import sweedit'",
        ],
        user="root",
    )
    if prewarm.exit_code != 0:
        logger.warning(f"sweedit pre-warm failed (continuing): {prewarm.output.decode()[:500]}")

    # 4. Persist env vars in the container so `uv run` subshells inherit them.
    logger.info("Setting environment variables...")
    set_environment_variables(container, env_vars or {})

    return container, logger


def set_environment_variables(container, env_vars):
    for key, value in env_vars.items():
        value_literal = shlex.quote(str(value))
        export_line = f"export {key}={value_literal}"
        container.exec_run(["bash", "-c", export_line])
        container.exec_run(["bash", "-c", f"printf '%s\\n' {shlex.quote(export_line)} >> ~/.bashrc"])


# ---------------------------------------------------------------------------
# LLM credential mapping (local .env  ->  SWE-Edit env scheme)
# ---------------------------------------------------------------------------

_CLOUDGPT_BASE_URL = "https://cloudgpt-openai.azure-api.net/openai/v1/"


def _find_parent_repo_root() -> str:
    """Walk up from this file until we find the parent SWE-Chain repo root.

    This harness lives inside the SWE-Edit submodule
    (``<root>/evaluation/SWE-Edit/evaluation/swechain/utils.py``); ``cloudgpt_aoai.py``
    and ``continuous-swe/`` live in the parent repo root. Search upward rather than
    hard-coding a fixed number of ``dirname`` hops so the depth can change freely.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    while True:
        if os.path.exists(os.path.join(cur, "cloudgpt_aoai.py")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:  # reached filesystem root
            return here
        cur = parent


def _get_cloudgpt_token(api_config: dict[str, Any]) -> str:
    import sys

    sys.path.insert(0, _find_parent_repo_root())
    from cloudgpt_aoai import get_openai_token_provider

    token_provider = get_openai_token_provider(
        use_azure_cli=api_config.get("USE_AZURE_CLI"),
        use_device_code=api_config.get("USE_DEVICE_CODE"),
    )
    return token_provider()


def resolve_sweedit_llm_env(api_config: dict[str, Any]) -> dict[str, str]:
    """Convert one local API config into the env vars SWE-Edit's main agent reads
    in-container (``API_TYPE`` / ``OPENAI_API_KEY`` / ``MODEL`` / ...).

    CLOUDGPT has no native SWE-Edit support, so it's mapped onto the OpenAI path
    with a freshly-minted bearer token and the CloudGPT base URL.
    """
    api_type = api_config.get("API_TYPE")

    if api_type == "CLOUDGPT":
        model = api_config.get("MODEL", "")
        if not model:
            raise ValueError("MODEL is required for CLOUDGPT")
        return {
            "API_TYPE": "OPENAI",
            "OPENAI_API_KEY": _get_cloudgpt_token(api_config),
            "OPENAI_BASE_URL": _CLOUDGPT_BASE_URL,
            "MODEL": model,
        }

    # OPENAI / ANTHROPIC / AZURE_OPENAI configs already use SWE-Edit's key names.
    if api_type in ("OPENAI", "ANTHROPIC", "AZURE_OPENAI"):
        return {k: str(v) for k, v in api_config.items()}

    raise ValueError(f"Unknown API type: {api_type}")


def _editor_env_from_main(main_env: dict[str, str], editor_model: str | None) -> dict[str, str]:
    """Build ``LLM_EDITOR_*`` env vars (used by the viewer + editor subagents)
    from the resolved main-agent env, optionally overriding the model.

    SWE-Edit's viewer subagent reuses the editor config in-container, so this one
    block covers both ``llm-file-editor``, ``llm-editor`` and ``baseline-v2``.
    """
    api_type = main_env["API_TYPE"]
    out: dict[str, str] = {"LLM_EDITOR_API_TYPE": api_type}

    if api_type == "OPENAI":
        if key := main_env.get("OPENAI_API_KEY"):
            out["LLM_EDITOR_OPENAI_API_KEY"] = key
        if url := main_env.get("OPENAI_BASE_URL"):
            out["LLM_EDITOR_OPENAI_BASE_URL"] = url
        out["LLM_EDITOR_MODEL"] = editor_model or main_env.get("MODEL", "")
    elif api_type == "ANTHROPIC":
        if key := main_env.get("ANTHROPIC_API_KEY"):
            out["LLM_EDITOR_ANTHROPIC_API_KEY"] = key
        out["LLM_EDITOR_MODEL"] = editor_model or main_env.get("MODEL", "")
    elif api_type == "AZURE_OPENAI":
        if key := main_env.get("AZURE_API_KEY"):
            out["LLM_EDITOR_AZURE_API_KEY"] = key
        if endpoint := main_env.get("AZURE_ENDPOINT"):
            out["LLM_EDITOR_AZURE_ENDPOINT"] = endpoint
        out["LLM_EDITOR_DEPLOYMENT"] = editor_model or main_env.get("DEPLOYMENT", "")
        if version := main_env.get("AZURE_API_VERSION"):
            out["LLM_EDITOR_AZURE_API_VERSION"] = version
    else:
        raise ValueError(f"Unknown API type for editor: {api_type}")

    return out


def refresh_cloudgpt_token(env_vars: dict[str, str], api_config: dict[str, str]) -> None:
    """Refresh the CloudGPT bearer token in env_vars if the config uses CloudGPT.

    Call before each agent run so the token hasn't expired mid-sequence.
    """
    if api_config.get("API_TYPE") != "CLOUDGPT":
        return
    token = _get_cloudgpt_token(api_config)
    env_vars["OPENAI_API_KEY"] = token
    if "LLM_EDITOR_API_TYPE" in env_vars and env_vars["LLM_EDITOR_API_TYPE"] == "OPENAI":
        env_vars["LLM_EDITOR_OPENAI_API_KEY"] = token


def build_agent_env(
    api_config: dict[str, str],
    agent_type: str,
    editor_model: str | None,
) -> dict[str, str]:
    """Assemble the full env dict handed to the in-container SWE-Edit agent.

    The main-agent credentials are always present; ``LLM_EDITOR_*`` are added for
    the agent types that spawn a viewer/editor subagent.
    """
    env_vars = resolve_sweedit_llm_env(api_config)
    if agent_type in ("llm-file-editor", "llm-editor", "baseline-v2"):
        env_vars.update(_editor_env_from_main(env_vars, editor_model))
    return env_vars


def cleanup_container_and_logger(client, container, logger):
    cleanup_container(client, container, logger)
    close_logger(logger)
