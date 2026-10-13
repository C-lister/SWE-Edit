"""Roll out the SWE-Edit agent on the continuous SWE-bench dataset.

Mirrors ``evaluation/OpenHands/main.py`` (same three modes — sequential,
sequential-memory, interleaved — same per-sequence container + git-diff capture +
output shape) but drives the SWE-Edit agent (https://github.com/microsoft/SWE-Edit)
inside the container instead of the OpenHands CLI.

This lives inside the SWE-Edit submodule (``evaluation/swechain/``). Build the
sweedit wheel once from the submodule root::

    cd evaluation/SWE-Edit && uv build      # produces dist/sweedit-<ver>-py3-none-any.whl

The harness auto-detects that wheel; run main.py from the parent SWE-Chain repo
root. See README.md for full usage.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shlex
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from utils import (
    apply_ground_truth_patch,
    build_agent_env,
    build_swebench_container,
    cleanup_container_and_logger,
    commit_agent_changes,
    exec_run_with_timeout,
    get_git_diff_from_commit,
    load_api_configs,
    load_continuous_dataset,
    refresh_cloudgpt_token,
    setup_global_logger,
)

# Global logger instance
glogger: logging.Logger = None


def detect_repo_dir(client, image: str, test_cmds: str, source: str) -> str:
    """Return the repo dir inside the container. Mirrors filter.py: prefer an
    explicit ``cd /X`` in test_cmds, then the image's WORKDIR, then a swebench-pro
    fallback to ``/app``, else ``/testbed``."""
    m = re.search(r"cd\s+(/[^\s&|;]+)", test_cmds or "")
    if m:
        return m.group(1)
    try:
        wd = client.images.get(image).attrs.get("Config", {}).get("WorkingDir")
        if wd:
            return wd
    except Exception:  # noqa: BLE001
        pass
    if (source or "").lower() == "swebench-pro":
        return "/app"
    return "/testbed"


def build_prompt_instance(bug_fix: dict, sequence: dict) -> dict:
    """Build the dict the prompt template renders against — pulled entirely from
    the dataset, no Hugging Face lookups."""
    return {
        "instance_id": bug_fix["swebench_instance_id"],
        "repo": sequence.get("repo", ""),
        "base_commit": sequence.get("base_commit", ""),
        "problem_statement": bug_fix.get("problem_statement", ""),
        "FAIL_TO_PASS": bug_fix.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": bug_fix.get("PASS_TO_PASS", []),
    }


def render_prompt(template, instance: dict, workspace_path: str) -> str:
    instance["repo_path"] = workspace_path
    return template.render(instance=instance)


def build_trajectory_id(continuous_id: str, bug_order: int, bug_id: str) -> str:
    """Build a stable, bounded trajectory / session name for one agent run."""
    raw_id = f"{continuous_id}__{bug_order}__{bug_id}"
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_id)
    if len(safe_id) <= 180:
        return safe_id
    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe_id[:160]}__{digest}"


def run_sweedit_in_container(
    container,
    instance_logger,
    workspace_path: str,
    task_prompt: str,
    env_vars: dict,
    args,
    timeout: int,
    memory_file: str | None = None,
) -> None:
    """Run one SWE-Edit agent invocation inside the container on ``workspace_path``."""
    env_vars = dict(env_vars)
    env_vars["TASK_PROMPT"] = task_prompt
    env_vars["WORKDIR"] = workspace_path
    env_vars["REASONING_EFFORT"] = args.reasoning_effort
    env_vars["AGENT_TYPE"] = args.swedit_agent_type
    env_vars["SAVE_DIR"] = args.save_dir
    if args.enable_context_summarization:
        env_vars["ENABLE_CONTEXT_SUMMARIZATION"] = "1"
        env_vars["SUMMARIZATION_THRESHOLD_TOKENS"] = str(args.summarization_threshold_tokens)
        env_vars["SUMMARIZATION_KEEP_RECENT_MESSAGES"] = str(args.summarization_keep_recent_messages)
    else:
        env_vars.pop("ENABLE_CONTEXT_SUMMARIZATION", None)
    if memory_file:
        env_vars["MEMORY_FILE"] = memory_file
    else:
        env_vars.pop("MEMORY_FILE", None)

    wheel_in_container = f"/agent/{Path(args.whl_file_path).name}"
    agent_cmd = (
        f"cd {shlex.quote(workspace_path)} && "
        f"uv run --python 3.12 --with {shlex.quote(wheel_in_container)} /agent/run_agent.py"
    )

    instance_logger.info(f"Running SWE-Edit ({args.agent_type}) in {workspace_path}...")
    exec_result = exec_run_with_timeout(
        container,
        ["bash", "-c", agent_cmd],
        timeout=timeout,
        logger=instance_logger,
        user="root",
        environment=env_vars,
    )

    if exec_result.exit_code != 0:
        instance_logger.error(
            f"SWE-Edit failed (exit {exec_result.exit_code}): "
            f"{exec_result.output.decode('utf-8', errors='replace')[:800]}"
        )
    else:
        instance_logger.info("SWE-Edit completed successfully")


# ---------------------------------------------------------------------------
# Mode 1: Sequential — fix tasks one after another (cumulative repo state).
#   sequential-memory additionally preserves the agent's conversation context
#   across bugs via a shared MEMORY_FILE inside the container.
# ---------------------------------------------------------------------------


def run_sequential(client, sequence, template, args, traj_mount, api_config) -> dict:
    global glogger

    repo = sequence["repo"]
    base_commit = sequence["base_commit"]
    continuous_id = sequence["continuous_id"]
    bug_fixes = sequence["bug_fixes"]
    docker_image = sequence["docker_image"]
    preserve_context = args.mode == "sequential-memory"
    mode_label = "sequential-memory" if preserve_context else "sequential"

    safe_cid = re.sub(r"[^A-Za-z0-9_.-]", "_", continuous_id)
    memory_file = f"/tmp/sweedit-memory-{safe_cid}.jsonl" if preserve_context else None

    glogger.info(f"[{mode_label}] Starting {continuous_id} ({len(bug_fixes)} bugs)")

    repo_name = repo.split("/")[-1]
    workspace_path = f"/workspace/{repo_name}/"

    container = None
    instance_logger = None
    results = []

    try:
        env_vars = build_agent_env(api_config, args.swedit_agent_type, args.editor_model)

        container, instance_logger = build_swebench_container(
            client,
            docker_image=docker_image,
            whl_file_path=args.whl_file_path,
            agent_script_path=args.agent_script_path,
            run_id=f"cont-seqmem-{continuous_id}" if preserve_context else f"cont-seq-{continuous_id}",
            instance_id=bug_fixes[0]["swebench_instance_id"],
            env_vars=env_vars,
            mount=traj_mount,
        )

        repo_dir = detect_repo_dir(
            client, docker_image, bug_fixes[0].get("test_cmds", ""), sequence.get("source")
        )
        glogger.info(f"[{mode_label}] repo_dir={repo_dir}")

        exec_result = container.exec_run(
            ["bash", "-c", f"mkdir -p {workspace_path} && cp -r {repo_dir}/. {workspace_path}/"]
        )
        if exec_result.exit_code != 0:
            raise Exception(f"Workspace setup failed: {exec_result.output.decode()}")
        container.exec_run([
            "bash", "-c",
            f"cd {workspace_path} && (git reset --hard {base_commit} || true) && git clean -fdx",
        ])
        baseline = container.exec_run(["bash", "-c", f"cd {workspace_path} && git rev-parse HEAD"])
        prev_head = baseline.output.decode().strip() if baseline.exit_code == 0 else base_commit
        glogger.info(f"[{mode_label}] baseline HEAD={prev_head[:12]}")

        for i, bug_fix in enumerate(bug_fixes):
            bug_id = bug_fix["swebench_instance_id"]
            glogger.info(f"[{mode_label}] Bug {i+1}/{len(bug_fixes)}: {bug_id}")

            prompt_instance = build_prompt_instance(bug_fix, sequence)
            task_prompt = render_prompt(template, prompt_instance, workspace_path)
            env_vars["INSTANCE_ID"] = bug_id
            env_vars["SESSION_ID"] = build_trajectory_id(continuous_id, i + 1, bug_id)
            try:
                refresh_cloudgpt_token(env_vars, api_config)
                run_sweedit_in_container(
                    container, instance_logger, workspace_path,
                    task_prompt, env_vars, args, args.timeout,
                    memory_file=memory_file,
                )
            except TimeoutError:
                glogger.warning(f"[{mode_label}] Timeout on {bug_id}")

            new_head = commit_agent_changes(
                container, workspace_path, f"agent-fix-{i+1}-{bug_id}", instance_logger
            )
            model_patch = get_git_diff_from_commit(
                container, workspace_path, prev_head, instance_logger
            )
            prev_head = new_head

            results.append({
                "instance_id": bug_id,
                "model_name_or_path": args.model_name_or_path,
                "model_patch": model_patch,
                "continuous_id": continuous_id,
                "mode": mode_label,
                "bug_order": i + 1,
            })

        return {"continuous_id": continuous_id, "mode": mode_label, "results": results}

    except Exception as e:
        glogger.error(f"[{mode_label}] Error in {continuous_id}: {e}")
        glogger.error(traceback.format_exc())
        raise
    finally:
        if container and instance_logger:
            cleanup_container_and_logger(client, container, instance_logger)


# ---------------------------------------------------------------------------
# Mode 2: Interleaved — task → ground-truth state → task.
# ---------------------------------------------------------------------------


def run_interleaved(client, sequence, template, args, traj_mount, api_config) -> dict:
    global glogger

    repo = sequence["repo"]
    base_commit = sequence["base_commit"]
    continuous_id = sequence["continuous_id"]
    bug_fixes = sequence["bug_fixes"]
    docker_image = sequence["docker_image"]
    glogger.info(f"[interleaved] Starting {continuous_id}")

    repo_name = repo.split("/")[-1]
    workspace_path = f"/workspace/{repo_name}/"

    container = None
    instance_logger = None
    results = []

    try:
        env_vars = build_agent_env(api_config, args.swedit_agent_type, args.editor_model)

        container, instance_logger = build_swebench_container(
            client,
            docker_image=docker_image,
            whl_file_path=args.whl_file_path,
            agent_script_path=args.agent_script_path,
            run_id=f"cont-intlv-{continuous_id}",
            instance_id=bug_fixes[0]["swebench_instance_id"],
            env_vars=env_vars,
            mount=traj_mount,
        )

        repo_dir = detect_repo_dir(
            client, docker_image, bug_fixes[0].get("test_cmds", ""), sequence.get("source")
        )
        glogger.info(f"[interleaved] repo_dir={repo_dir}")

        exec_result = container.exec_run(
            ["bash", "-c", f"mkdir -p {workspace_path} && cp -r {repo_dir}/. {workspace_path}/"]
        )
        if exec_result.exit_code != 0:
            raise Exception(f"Workspace setup failed: {exec_result.output.decode()}")

        for i, bug_fix in enumerate(bug_fixes):
            bug_id = bug_fix["swebench_instance_id"]
            glogger.info(f"[interleaved] Bug {i+1}/{len(bug_fixes)}: {bug_id}")
            container.exec_run([
                "bash", "-c",
                f"cd {workspace_path} && (git reset --hard {base_commit} || true) && git clean -fdx",
            ])
            for j in range(i):
                prior_bug = bug_fixes[j]
                prior_id = prior_bug["swebench_instance_id"]
                test_patch = prior_bug.get("test_patch", "")
                if test_patch:
                    glogger.info(f"[interleaved] Applying test patch {j+1}/{i} for bug {prior_id}")
                    if not apply_ground_truth_patch(
                        container, workspace_path, test_patch, f"test-{prior_id}", instance_logger
                    ):
                        glogger.warning(
                            f"[interleaved] Ground-truth test patch failed for {prior_id}, continuing"
                        )

                gold_patch = prior_bug.get("patch", "")
                if not gold_patch:
                    continue
                glogger.info(f"[interleaved] Applying gold patch {j+1}/{i} for bug {prior_id}")
                if not apply_ground_truth_patch(
                    container, workspace_path, gold_patch, f"gold-{prior_id}", instance_logger
                ):
                    glogger.warning(f"[interleaved] Ground-truth patch failed for {prior_id}, continuing")

            baseline = container.exec_run(["bash", "-c", f"cd {workspace_path} && git rev-parse HEAD"])
            prev_head = baseline.output.decode().strip() if baseline.exit_code == 0 else base_commit
            glogger.info(f"[interleaved] baseline HEAD={prev_head[:12]}")

            prompt_instance = build_prompt_instance(bug_fix, sequence)
            task_prompt = render_prompt(template, prompt_instance, workspace_path)
            env_vars["INSTANCE_ID"] = bug_id
            env_vars["SESSION_ID"] = build_trajectory_id(continuous_id, i + 1, bug_id)
            try:
                refresh_cloudgpt_token(env_vars, api_config)
                run_sweedit_in_container(
                    container, instance_logger, workspace_path,
                    task_prompt, env_vars, args, args.timeout,
                    memory_file=None,
                )
            except TimeoutError:
                glogger.warning(f"[interleaved] Timeout on {bug_id}")

            commit_agent_changes(
                container, workspace_path, f"agent-fix-{i+1}-{bug_id}", instance_logger
            )
            model_patch = get_git_diff_from_commit(
                container, workspace_path, prev_head, instance_logger
            )

            results.append({
                "instance_id": bug_id,
                "model_name_or_path": args.model_name_or_path,
                "model_patch": model_patch,
                "continuous_id": continuous_id,
                "mode": "interleaved",
                "bug_order": i + 1,
            })

        return {"continuous_id": continuous_id, "mode": "interleaved", "results": results}

    except Exception as e:
        glogger.error(f"[interleaved] Error in {continuous_id}: {e}")
        glogger.error(traceback.format_exc())
        raise
    finally:
        if container and instance_logger:
            cleanup_container_and_logger(client, container, instance_logger)


# ---------------------------------------------------------------------------
# Output adaptation — same shape as OpenHands so evaluate.py consumes it as-is.
# ---------------------------------------------------------------------------


_SEQ_META_KEYS = ("continuous_id", "repo", "base_commit", "docker_image", "source", "date_range")
_BUG_KEEP_KEYS = ("swebench_instance_id", "patch", "test_patch", "test_cmds",
                  "FAIL_TO_PASS", "PASS_TO_PASS")


def merge_agent_into_sequence(sequence: dict, agent_result: dict, args) -> dict:
    """Merge per-bug agent patches back into the input sequence shape."""
    patch_by_iid = {
        r["instance_id"]: r.get("model_patch", "")
        for r in agent_result.get("results", [])
    }
    out_bugs = []
    for bug in sequence.get("bug_fixes", []):
        iid = bug["swebench_instance_id"]
        merged = {k: bug[k] for k in _BUG_KEEP_KEYS if k in bug}
        merged["model_patch"] = patch_by_iid.get(iid, "")
        out_bugs.append(merged)

    out = {k: sequence[k] for k in _SEQ_META_KEYS if k in sequence}
    out["bug_fixes"] = out_bugs
    out["mode"] = agent_result.get("mode")
    out["model_name_or_path"] = args.model_name_or_path
    return out


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Continuous SWE-bench rollout for SWE-Edit")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["sequential", "sequential-memory", "interleaved"],
        help="Evaluation mode. sequential-memory also preserves the agent's "
             "conversation context across bugs.",
    )
    parser.add_argument(
        "--dataset", type=str, default="continuous-swe/data/data.jsonl",
        help="Path to the continuous SWE-bench dataset JSONL (relative to repo root)",
    )
    parser.add_argument(
        "--local_traj_path", type=str, default=None,
        help="Local path for agent trajectories (mounted at /tmp/.traj). "
             "Defaults to <cwd>/trajectory/<agent_type>/<model_name>/<mode>.",
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Model name used to namespace trajectories + outputs "
             "(<cwd>/trajectory/<agent_type>/<model_name>/<mode> and "
             "<cwd>/outputs/<agent_type>/<model_name>/<mode>.jsonl). Defaults to the MODEL "
             "from your API config.",
    )
    parser.add_argument(
        "--whl_file_path", type=str, default=None,
        help="Path to the sweedit wheel. Defaults to the newest "
             "<submodule>/dist/sweedit-*.whl (run `uv build` in the SWE-Edit submodule first).",
    )
    parser.add_argument(
        "--agent_script_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "run_agent.py"),
        help="Path to the in-container agent script",
    )
    parser.add_argument(
        "--agent_type", type=str, default="baseline",
        choices=["baseline", "summarize", "llm-editor"],
        help="Agent variant: 'baseline' (plain agent), 'summarize' (baseline + "
             "context summarization), or 'llm-editor' (the full SWE-Edit with "
             "viewer + editor subagents, no summarization).",
    )
    parser.add_argument(
        "--editor_model", type=str, default=None,
        help="Model for the viewer/editor subagent (LLM_EDITOR_MODEL). Defaults "
             "to the main agent's model. Used by the 'llm-editor' agent type.",
    )
    parser.add_argument("--prompt_template", type=str, default="prompt.j2", help="Prompt template file name")
    parser.add_argument(
        "--reasoning_effort", type=str, default="medium",
        choices=["low", "medium", "high"], help="Reasoning effort level",
    )
    parser.add_argument(
        "--summarization_threshold_tokens", type=int, default=50000,
        help="For the 'summarize' agent type: context size (input tokens) "
             "above which summarization triggers.",
    )
    parser.add_argument(
        "--summarization_keep_recent_messages", type=int, default=50,
        help="Number of most recent messages kept verbatim when summarizing.",
    )
    parser.add_argument("--max_workers", type=int, default=8, help="Max sequences in parallel")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-agent-run timeout (seconds)")
    parser.add_argument("--sequence_ids", type=str, nargs="*", default=None,
                        help="Optional: only run specific continuous_id(s)")
    parser.add_argument("--max_sequences", type=int, default=None, help="Max sequences to process")
    args = parser.parse_args()

    # Map the high-level agent type (used for path namespacing + logging) onto the
    # underlying SWE-Edit variant the container runs and whether to summarize.
    _AGENT_TYPE_SPEC = {
        "baseline":   {"swedit_type": "baseline",   "summarize": False},
        "summarize":  {"swedit_type": "baseline",   "summarize": True},
        "llm-editor": {"swedit_type": "llm-editor", "summarize": False},
    }
    spec = _AGENT_TYPE_SPEC[args.agent_type]
    args.swedit_agent_type = spec["swedit_type"]
    args.enable_context_summarization = spec["summarize"]

    if args.agent_type == "llm-editor" and not args.editor_model:
        glogger_warn = (
            "No --editor_model provided; the viewer/editor subagent will reuse the "
            "main agent's model. Pass --editor_model for the intended (smaller) editor."
        )
    else:
        glogger_warn = None

    glogger = setup_global_logger()
    glogger.info("=" * 60)
    glogger.info(f"Continuous SWE-Edit Evaluation — Mode: {args.mode}")
    glogger.info("=" * 60)
    if glogger_warn:
        glogger.warning(glogger_warn)

    # Load API configs up front so the model name can namespace outputs.
    api_configs = load_api_configs()
    if not api_configs:
        raise ValueError("No API configurations found in environment variables")
    glogger.info(f"Loaded {len(api_configs)} API configuration(s)")

    # Model name (used to namespace trajectories + outputs) — from --model_name,
    # else the first API config's MODEL/DEPLOYMENT. Sanitized for use as a path.
    raw_model = args.model_name or api_configs[0].get("MODEL") or api_configs[0].get("DEPLOYMENT") or "model"
    model_name = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_model)
    args.model_name = model_name
    args.model_name_or_path = model_name

    # Trajectories -> <cwd>/trajectory/<agent_type>/<model_name>/<mode> (mounted at /tmp/.traj);
    # outputs -> <cwd>/outputs/<agent_type>/<model_name>/<mode>.jsonl.
    cwd = os.getcwd()
    if not args.local_traj_path:
        args.local_traj_path = os.path.join(cwd, "trajectory", args.agent_type, model_name, args.mode)
    args.local_traj_path = os.path.abspath(args.local_traj_path)
    os.makedirs(args.local_traj_path, exist_ok=True)
    args.save_dir = "/tmp/.traj"
    args.output_file = os.path.join(cwd, "outputs", args.agent_type, model_name, f"{args.mode}.jsonl")
    args.agent_script_path = os.path.abspath(args.agent_script_path)

    # Default the wheel to the newest <submodule>/dist/sweedit-*.whl. The submodule
    # root is two levels up from this file (swechain -> evaluation -> SWE-Edit).
    if not args.whl_file_path:
        submodule_root = Path(__file__).resolve().parents[2]
        candidates = sorted(
            (submodule_root / "dist").glob("sweedit-*.whl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No sweedit wheel found in {submodule_root / 'dist'}. Build it first: "
                f"`cd {submodule_root} && uv build`, or pass --whl_file_path explicitly."
            )
        args.whl_file_path = str(candidates[0])
        glogger.info(f"Auto-detected sweedit wheel: {args.whl_file_path}")
    args.whl_file_path = os.path.abspath(args.whl_file_path)
    if not os.path.exists(args.whl_file_path):
        raise FileNotFoundError(
            f"sweedit wheel not found: {args.whl_file_path}. Build it with `uv build` "
            "in the SWE-Edit submodule and pass --whl_file_path dist/sweedit-<ver>-py3-none-any.whl"
        )
    output_file = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    client = docker.from_env(timeout=600)

    dataset = load_continuous_dataset(os.path.abspath(args.dataset))
    if args.sequence_ids:
        dataset = [s for s in dataset if s["continuous_id"] in args.sequence_ids]
        glogger.info(f"Filtered to {len(dataset)} sequences by ID")
    if args.max_sequences:
        dataset = dataset[: args.max_sequences]
    glogger.info(f"Processing {len(dataset)} sequences")

    template_dir = os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(args.prompt_template)

    mode_fn = {
        "sequential": run_sequential,
        "sequential-memory": run_sequential,
        "interleaved": run_interleaved,
    }[args.mode]

    traj_mount = {"local_path": args.local_traj_path, "container_path": "/tmp/.traj"}

    glogger.info("Configuration:")
    glogger.info(f"  Mode: {args.mode}")
    glogger.info(f"  Model name: {args.model_name}")
    glogger.info(f"  Trajectories: {args.local_traj_path}")
    glogger.info(f"  Agent type: {args.agent_type}")
    glogger.info(f"  Wheel: {args.whl_file_path}")
    glogger.info(f"  Sequences: {len(dataset)}")
    glogger.info(f"  Max workers: {args.max_workers}")
    glogger.info(f"  Timeout: {args.timeout}s")
    glogger.info(f"  Output: {output_file}")

    # Resume: skip sequences already present in the output file.
    completed_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "continuous_id" in data:
                        completed_ids.add(data["continuous_id"])
                except json.JSONDecodeError:
                    pass
    if completed_ids:
        before_count = len(dataset)
        dataset = [s for s in dataset if s["continuous_id"] not in completed_ids]
        skipped = before_count - len(dataset)
        if skipped > 0:
            glogger.info(f"Resuming run: skipped {skipped} already-completed sequences.")
            glogger.info(f"Sequences remaining: {len(dataset)}")
    if not dataset:
        glogger.info("All sequences already processed. Exiting.")
        sys.exit(0)

    completed = 0
    failed = 0

    if args.max_workers == 1:
        for idx, sequence in enumerate(dataset):
            cid = sequence["continuous_id"]
            try:
                result = mode_fn(
                    client, sequence, template, args, traj_mount,
                    api_configs[idx % len(api_configs)],
                )
                merged = merge_agent_into_sequence(sequence, result, args)
                with open(output_file, "a") as f:
                    f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                completed += 1
                glogger.info(f"[{completed}/{len(dataset)}] Completed {cid}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                glogger.error(f"Failed {cid}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for idx, sequence in enumerate(dataset):
                future = executor.submit(
                    mode_fn,
                    client, sequence, template, args, traj_mount,
                    api_configs[idx % len(api_configs)],
                )
                futures[future] = sequence
            for future in as_completed(futures):
                sequence = futures[future]
                cid = sequence["continuous_id"]
                try:
                    result = future.result()
                    merged = merge_agent_into_sequence(sequence, result, args)
                    with open(output_file, "a") as f:
                        f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                    completed += 1
                    glogger.info(f"[{completed}/{len(dataset)}] Completed {cid}")
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    glogger.error(f"Failed {cid}: {e}")

    glogger.info("=" * 60)
    glogger.info("Evaluation Complete!")
    glogger.info(f"  Successful: {completed}")
    glogger.info(f"  Failed: {failed}")
    glogger.info(f"  Output: {output_file}")
    glogger.info("=" * 60)
