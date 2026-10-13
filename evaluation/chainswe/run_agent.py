# Runs inside the SWE-bench container via:
#   uv run --python 3.12 --with /agent/sweedit-<ver>-py3-none-any.whl /agent/run_agent.py
#
# Adapted from SWE-Edit's evaluation/swebench-verified/run_agent.py, with one
# addition: optional cross-call conversation memory (for `sequential-memory`
# mode). SWE-Edit has no built-in conversation resume, so when MEMORY_FILE is
# set we seed the agent's context with the messages persisted by the previous
# bug in the sequence, then dump the cumulative context back out afterwards.

import asyncio
import json
import os
from pathlib import Path


def _load_memory(agent, memory_file: str) -> None:
    """Seed the agent's context with prior-bug messages, if any."""
    path = Path(memory_file)
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        messages = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:  # corrupt / unreadable -> start fresh
        print(f"[run_agent] could not load memory from {memory_file}: {e}")
        return
    if isinstance(messages, list) and messages:
        # ContextManager stores the running conversation as a plain list of
        # message dicts; replacing it preserves full prior context.
        agent.context._messages = messages
        print(f"[run_agent] seeded {len(messages)} prior messages from {memory_file}")


def _dump_memory(agent, memory_file: str) -> None:
    """Persist the cumulative context so the next bug can resume it."""
    path = Path(memory_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(agent.context._messages, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[run_agent] saved {len(agent.context._messages)} messages to {memory_file}")


def main() -> None:
    from sweedit.core.agenthub import (
        SwebenchAgent,
        SwebenchAgentV2,
        SwebenchAgentWithLLMEditor,
        SwebenchAgentWithLLMFileEditor,
    )

    task_prompt = os.environ.get("TASK_PROMPT")
    if not task_prompt:
        raise ValueError("TASK_PROMPT environment variable not set")

    session_id = os.environ.get("SESSION_ID") or os.environ.get("INSTANCE_ID")
    workdir = os.environ.get("WORKDIR")
    reasoning_effort = os.environ.get("REASONING_EFFORT", "medium")
    save_dir = os.environ.get("SAVE_DIR", ".traj/")
    agent_type = os.environ.get("AGENT_TYPE", "baseline")
    memory_file = os.environ.get("MEMORY_FILE")

    # Optional context summarization (compacts old turns to stay within the
    # context window). Enabled via ENABLE_CONTEXT_SUMMARIZATION; tuned via the
    # SUMMARIZATION_* vars.
    enable_summarization = os.environ.get("ENABLE_CONTEXT_SUMMARIZATION", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    summarization_threshold = int(os.environ.get("SUMMARIZATION_THRESHOLD_TOKENS", "50000"))
    summarization_keep_recent = int(os.environ.get("SUMMARIZATION_KEEP_RECENT_MESSAGES", "50"))

    summarization_kwargs = {
        "enable_context_summarization": enable_summarization,
        "summarization_threshold_tokens": summarization_threshold,
        "summarization_keep_recent_messages": summarization_keep_recent,
    }

    if agent_type == "baseline":
        agent = SwebenchAgent(
            session_id=session_id, reasoning_effort=reasoning_effort, save_dir=save_dir, **summarization_kwargs
        )
    elif agent_type == "baseline-v2":
        agent = SwebenchAgentV2(
            session_id=session_id, reasoning_effort=reasoning_effort, save_dir=save_dir, **summarization_kwargs
        )
    elif agent_type == "llm-file-editor":
        agent = SwebenchAgentWithLLMFileEditor(
            session_id=session_id, reasoning_effort=reasoning_effort, save_dir=save_dir, **summarization_kwargs
        )
    elif agent_type == "llm-editor":
        agent = SwebenchAgentWithLLMEditor(
            session_id=session_id, reasoning_effort=reasoning_effort, save_dir=save_dir, **summarization_kwargs
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if enable_summarization:
        print(
            f"[run_agent] context summarization enabled "
            f"(threshold={summarization_threshold} tokens, keep_recent={summarization_keep_recent} messages)"
        )

    agent.set_workdir(workdir)

    if memory_file:
        _load_memory(agent, memory_file)

    asyncio.run(agent.run(user_message=task_prompt))

    if memory_file:
        _dump_memory(agent, memory_file)


if __name__ == "__main__":
    main()
