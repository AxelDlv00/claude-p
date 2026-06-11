#!/usr/bin/env python3
"""Claude Code interactive-TUI backend with `claude -p` compatible output.

This script does not invoke `claude -p`. It starts interactive `claude` under a
pseudo-TTY, captures the rendered terminal, extracts the assistant answer, and
emits text/json/stream-json output shaped like `claude -p`.

Compatibility target:
- Same line-oriented JSON transport.
- Same core event families: system init, stream_event message_start,
  content_block_start/delta/stop, assistant, message_delta, message_stop, result.
- Usage/cost/tool events are best-effort placeholders because the interactive
  TUI does not expose a machine-readable protocol.
"""

from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
from pathlib import Path
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC_RE = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")
SPINNER_RE = re.compile(r"\n?[✳✶✻✽✢·].*$", re.DOTALL)
NON_TERMINAL_STOP_REASONS = {"tool_use", "pause_turn"}
SUBSCRIPTION_BACKEND_ENV_OVERRIDES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)
VARIADIC_PROMPT_ATTRS = (
    "tools",
    "allowed_tools",
    "disallowed_tools",
    "add_dir",
    "files",
    "mcp_config",
    "betas",
)


def warn(message: str) -> None:
    print(f"claude_tui_agent.py: warning: {message}", file=sys.stderr)


def append_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def append_value(cmd: list[str], flag: str, value: str | None) -> None:
    if value is not None:
        cmd.extend([flag, value])


def append_optional_value(cmd: list[str], flag: str, value: str | None) -> None:
    if value is None:
        return
    cmd.append(flag)
    if value:
        cmd.append(value)


def append_repeated_values(cmd: list[str], flag: str, values: list[str] | None) -> None:
    values = flatten_cli_values(values)
    if not values:
        return
    for value in values:
        cmd.extend([flag, value])


def append_variadic_values(cmd: list[str], flag: str, values: list[str] | None) -> None:
    values = flatten_cli_values(values)
    if not values:
        return
    cmd.append(flag)
    cmd.extend(values)


def now_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def emit(obj: dict, enabled: bool = True) -> None:
    if enabled:
        print(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), flush=True)


def clean_terminal(text: str) -> str:
    text = OSC_RE.sub("", text)
    text = ANSI_RE.sub("", text)
    return text.replace("\r", "").replace("\u00a0", " ")


def compact_for_detection(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_terminal(text).lower())


def flatten_cli_values(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    flattened: list[str] = []
    for value in values:
        if isinstance(value, str):
            flattened.append(value)
        elif isinstance(value, (list, tuple)):
            flattened.extend(str(item) for item in value)
        else:
            flattened.append(str(value))
    return flattened


def recover_prompt_from_variadic_args(args: argparse.Namespace) -> None:
    """Recover prompt-last invocations after argparse variadic option capture.

    Claude's CLI accepts options such as `--tools Bash Edit` as variadic lists,
    while this wrapper also accepts a free-form positional prompt. Argparse has
    no way to know where a variadic option ends when the prompt is last, so if
    no prompt was parsed and stdin is interactive, treat the final captured
    variadic token as the prompt.
    """
    if args.prompt is not None or not sys.stdin.isatty():
        return
    for attr in VARIADIC_PROMPT_ATTRS:
        values = flatten_cli_values(getattr(args, attr, None))
        if len(values) > 1:
            args.prompt = values.pop()
            setattr(args, attr, values)
            return


def normalize_answer(text: str) -> str:
    text = clean_terminal(text)
    text = SPINNER_RE.sub("", text)
    # Drop common TUI chrome if it leaked into the block.
    text = re.split(r"\n?────────────────", text, maxsplit=1)[0]
    return text.strip()


def extract_assistant_snapshot(transcript: str) -> str:
    clean = clean_terminal(transcript)
    # Claude Code prefixes each assistant turn with a bullet glyph, but the
    # exact glyph changed across versions: ⏺ (U+23FA) in older builds, ●
    # (U+25CF) in v2.1.x. Anchor on the last bullet of either kind — keying on
    # only the old glyph silently returns "" on current Claude Code, which
    # disables every terminal-scrape path (live deltas + the subagent fallback).
    marker = max(clean.rfind("⏺"), clean.rfind("●"))
    if marker < 0:
        return ""
    return normalize_answer(clean[marker + 1 :])


def classify_failure(transcript: str, assistant_text: str, timed_out: bool) -> str | None:
    interactive_block = classify_interactive_block(f"{transcript}\n{assistant_text}")
    if interactive_block:
        if assistant_text and interactive_block == "workspace_trust_blocked":
            return None
        return interactive_block
    if assistant_text:
        return None
    if timed_out:
        return "assistant_output_timeout"
    return "assistant_output_not_found"


def classify_interactive_block(text: str) -> str | None:
    low = clean_terminal(text).lower()
    compact = compact_for_detection(text)
    if "failed to authenticate" in low or "api error: 403" in low or "pleaserunlogin" in compact:
        return "auth_blocked"
    if "you've hit your limit" in low or "you have hit your limit" in low or "hit your limit" in low:
        return "rate_limit"
    if (
        ("do you trust" in low and "folder" in low)
        or "workspacetrust" in compact
        or ("quicksafetycheck" in compact and ("itrustthisfolder" in compact or "accessingworkspace" in compact))
    ):
        return "workspace_trust_blocked"
    if "permission" in low and ("allow" in low or "deny" in low):
        return "tool_approval_blocked"
    return None


def _iteration_from_api_usage(api_usage: dict) -> dict:
    return {
        "input_tokens": api_usage.get("input_tokens"),
        "output_tokens": api_usage.get("output_tokens"),
        "cache_read_input_tokens": api_usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": api_usage.get("cache_creation_input_tokens"),
        "cache_creation": {
            "ephemeral_5m_input_tokens": None,
            "ephemeral_1h_input_tokens": None,
        },
        "type": "message",
    }


def build_usage(output_text: str) -> dict:
    # Fallback when no JSONL data is available: shape-compatible with null values.
    approx_output_tokens = max(1, len(output_text.split()))
    return {
        "input_tokens": None,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "output_tokens": approx_output_tokens,
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        "service_tier": None,
        "cache_creation": {"ephemeral_1h_input_tokens": None, "ephemeral_5m_input_tokens": None},
        "iterations": [
            {
                "input_tokens": None,
                "output_tokens": approx_output_tokens,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": None,
                    "ephemeral_1h_input_tokens": None,
                },
                "type": "message",
            }
        ],
        "speed": None,
    }


def build_usage_from_events(events: list[dict]) -> dict:
    """Build a claude -p compatible usage object from all JSONL assistant events.

    Each assistant event in the JSONL corresponds to one turn (including
    intermediate tool-use turns). We sum across all turns for the totals and
    expose per-turn data in the iterations array, matching the shape that
    claude -p --output-format stream-json produces.
    """
    iterations = []
    total: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    for event in events:
        message = event.get("message", {})
        api_usage = message.get("usage")
        if not isinstance(api_usage, dict):
            continue
        for key in total:
            total[key] += api_usage.get(key) or 0
        iterations.append(_iteration_from_api_usage(api_usage))

    if not iterations:
        return build_usage("")

    return {
        "input_tokens": total["input_tokens"] or None,
        "cache_creation_input_tokens": total["cache_creation_input_tokens"] or None,
        "cache_read_input_tokens": total["cache_read_input_tokens"] or None,
        "output_tokens": total["output_tokens"] or None,
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        "service_tier": None,
        "cache_creation": {"ephemeral_1h_input_tokens": None, "ephemeral_5m_input_tokens": None},
        "iterations": iterations,
        "speed": None,
    }


def build_tui_env(args: argparse.Namespace) -> dict[str, str]:
    env = {**os.environ, "NO_COLOR": "1", "TERM": args.term}
    if not args.preserve_provider_env:
        for name in SUBSCRIPTION_BACKEND_ENV_OVERRIDES:
            env.pop(name, None)
    return env


def extract_text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def canonical_json_if_equivalent(left: str, right: str) -> str | None:
    try:
        left_obj = json.loads(left)
        right_obj = json.loads(right)
    except json.JSONDecodeError:
        return None
    if left_obj != right_obj:
        return None
    return json.dumps(right_obj, ensure_ascii=False, separators=(",", ":"))


def is_terminal_assistant_message(message: dict) -> bool:
    stop_reason = message.get("stop_reason")
    return stop_reason is not None and stop_reason not in NON_TERMINAL_STOP_REASONS


def _claude_config_dir() -> Path:
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(env) if env else Path.home() / ".claude"


# Per-session / per-instance state that two concurrent `claude` TUIs must NOT
# share. When a second claude runs on the same CLAUDE_CONFIG_DIR as a still-live
# first one, it fails to persist its session (only the early `ai-title` lands in
# `projects/**/<sid>.jsonl`), so --stream-session-events has nothing to tail and
# downstream logging is empty even though the run completes. Isolating just
# these — and symlinking everything else — fixes persistence with no heavy copy.
_ISOLATED_STATE = frozenset({
    "projects",        # session JSONL store — THE one that must be per-instance
    "sessions", "session-env", "ide", "shell-snapshots",
    "file-history", "history.jsonl", "tasks", "todos",
})


def _isolate_config_dir() -> None:
    """Point CLAUDE_CONFIG_DIR at a throwaway dir that symlinks the base config
    (auth, plugins, caches, settings — shared, so startup stays fast and
    authenticated) but keeps a FRESH session store, so this claude can run
    concurrently with another on the base dir without the session-persistence
    clash. Old sessions aren't preserved — the point is that new ones persist.
    Set before anything reads the config dir; cleaned up at exit."""
    base = _claude_config_dir()
    if not base.exists():
        return
    iso = Path(tempfile.mkdtemp(prefix="claude-p-cfg-"))
    for entry in base.iterdir():
        dst = iso / entry.name
        if entry.name in _ISOLATED_STATE:
            continue  # absent → claude creates a fresh per-instance copy
        try:
            if entry.name == ".claude.json":
                shutil.copy2(entry, dst)  # writable per-instance (onboarding/trust flags)
            else:
                os.symlink(entry, dst)    # auth / plugins / caches / settings shared (read)
        except OSError:
            pass
    os.environ["CLAUDE_CONFIG_DIR"] = str(iso)
    atexit.register(lambda: shutil.rmtree(iso, ignore_errors=True))


def _find_session_jsonl(session_id: str) -> Path | None:
    pattern = str(_claude_config_dir() / "projects" / "**" / f"{session_id}.jsonl")
    paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def read_all_assistant_events(session_id: str) -> list[dict]:
    """Read all assistant events from the session JSONL in order.

    Interactive Claude Code writes the same canonical JSONL as claude -p.
    Each assistant event covers one turn (text response or tool-use turn).
    Reading all of them lets us compute accurate multi-turn usage totals and
    a correct num_turns count.
    """
    path = _find_session_jsonl(session_id)
    if path is None:
        return []
    events: list[dict] = []
    try:
        with path.open() as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "assistant":
                    continue
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                events.append(event)
    except OSError:
        pass
    return events


def read_persisted_assistant(session_id: str, *, require_terminal: bool = False) -> dict | None:
    """Read Claude Code's persisted JSONL for the final assistant message.

    The interactive terminal is a lossy rendering surface: wide glyphs, cursor
    redraws, and spinner updates can drop or smear characters in the captured
    TTY transcript. Claude Code still writes the canonical session JSONL for
    interactive sessions. When available, use it as the source of truth for the
    final assistant message while keeping the TUI transcript as provenance.
    """
    path = _find_session_jsonl(session_id)
    if path is None:
        return None
    latest: dict | None = None
    try:
        with path.open() as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "assistant":
                    continue
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                text = extract_text_from_content(message.get("content"))
                if not text:
                    continue
                terminal = is_terminal_assistant_message(message)
                if require_terminal and not terminal:
                    continue
                latest = {
                    "path": str(path),
                    "text": text,
                    "message": message,
                    "model": message.get("model"),
                    "message_id": message.get("id"),
                    "usage": message.get("usage"),
                    "stop_reason": message.get("stop_reason"),
                    "terminal": terminal,
                }
    except OSError:
        return None
    return latest


def _drain_session_events(session_id: str, state: dict, stream_json: bool) -> None:
    """Echo new canonical assistant/user events from the session JSONL.

    Claude Code persists each turn to ``~/.claude/projects/**/<sid>.jsonl``
    in the same shape ``claude -p --output-format stream-json`` produces:
    ``assistant`` events carry text / thinking / tool_use content blocks,
    and ``user`` events carry tool_result blocks. Re-emitting each new line
    as it lands gives a downstream stream-json consumer FULL live logging —
    tool calls and their results, not just scraped terminal text.

    ``state`` is a caller-owned dict ({} initially) that tracks the located
    file and a byte offset, so each line is emitted exactly once across
    repeated calls. Only complete (newline-terminated) lines are emitted;
    a trailing partial line is left for the next drain.

    ``user`` events are emitted only when their ``message.content`` is a
    list (tool-result turns) — the initial human prompt is a string-content
    ``user`` event, which both isn't useful downstream and would break
    consumers that iterate ``content`` expecting blocks.
    """
    if state.get("path") is None:
        path = _find_session_jsonl(session_id)
        if path is None:
            return
        state["path"] = path
        state["offset"] = 0
    path = state["path"]
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= state.get("offset", 0):
        return
    try:
        with path.open("rb") as f:
            f.seek(state.get("offset", 0))
            chunk = f.read()
    except OSError:
        return
    last_nl = chunk.rfind(b"\n")
    if last_nl == -1:
        return
    complete = chunk[: last_nl + 1]
    state["offset"] = state.get("offset", 0) + len(complete)
    for raw_line in complete.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        if etype == "assistant":
            emit(event, enabled=stream_json)
            state["emitted"] = state.get("emitted", 0) + 1
        elif etype == "user" and isinstance(message.get("content"), list):
            emit(event, enabled=stream_json)
            state["emitted"] = state.get("emitted", 0) + 1


def run_tui(args: argparse.Namespace, stream_json: bool) -> tuple[str, str, int | None, bool, float]:
    cmd = ["claude", "--session-id", args.session_id]

    # Pass through options that the interactive `claude` entrypoint itself
    # understands. Print-only options are handled by this wrapper and are not
    # forwarded.
    append_variadic_values(cmd, "--add-dir", args.add_dir)
    append_value(cmd, "--agent", args.agent)
    append_value(cmd, "--agents", args.agents)
    append_flag(cmd, args.allow_dangerously_skip_permissions, "--allow-dangerously-skip-permissions")
    append_variadic_values(cmd, "--allowedTools", args.allowed_tools)
    append_value(cmd, "--append-system-prompt", args.append_system_prompt)
    append_variadic_values(cmd, "--betas", args.betas)
    append_flag(cmd, args.brief, "--brief")
    append_flag(cmd, args.chrome, "--chrome")
    append_flag(cmd, args.no_chrome, "--no-chrome")
    append_flag(cmd, args.continue_session, "--continue")
    append_flag(cmd, args.dangerously_skip_permissions, "--dangerously-skip-permissions")
    append_optional_value(cmd, "--debug", args.debug)
    append_value(cmd, "--debug-file", args.debug_file)
    append_flag(cmd, args.disable_slash_commands, "--disable-slash-commands")
    append_variadic_values(cmd, "--disallowedTools", args.disallowed_tools)
    append_value(cmd, "--effort", args.effort)
    append_flag(cmd, args.exclude_dynamic_system_prompt_sections, "--exclude-dynamic-system-prompt-sections")
    append_variadic_values(cmd, "--file", args.files)
    append_flag(cmd, args.fork_session, "--fork-session")
    append_optional_value(cmd, "--from-pr", args.from_pr)
    append_flag(cmd, args.ide, "--ide")
    append_value(cmd, "--json-schema", args.json_schema)
    append_variadic_values(cmd, "--mcp-config", args.mcp_config)
    append_flag(cmd, args.mcp_debug, "--mcp-debug")
    append_variadic_values(cmd, "--tools", args.tools)
    append_value(cmd, "--model", args.model)
    append_value(cmd, "--name", args.name)
    append_value(cmd, "--permission-mode", args.permission_mode)
    append_repeated_values(cmd, "--plugin-dir", args.plugin_dir)
    append_repeated_values(cmd, "--plugin-url", args.plugin_url)
    append_optional_value(cmd, "--remote-control", args.remote_control)
    append_value(cmd, "--remote-control-session-name-prefix", args.remote_control_session_name_prefix)
    append_optional_value(cmd, "--resume", args.resume)
    append_value(cmd, "--setting-sources", args.setting_sources)
    append_value(cmd, "--settings", args.settings)
    append_flag(cmd, args.strict_mcp_config, "--strict-mcp-config")
    append_value(cmd, "--system-prompt", args.system_prompt)
    append_optional_value(cmd, "--tmux", args.tmux)
    append_optional_value(cmd, "--worktree", args.worktree)

    cmd.append(args.prompt)
    master, slave = pty.openpty()
    env = build_tui_env(args)
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=args.cwd,
        env=env,
    )
    os.close(slave)

    raw = bytearray()
    last_output = time.time()
    last_snapshot = ""
    last_jsonl_poll = 0.0
    timed_out = True
    trust_sent = False
    # When claude-p is spawned inside another Claude Code (a dispatcher's Bash
    # tool launches it for a subagent), the parent injects
    # CLAUDE_CODE_CHILD_SESSION=1 into our environment — and that very flag is
    # why Claude Code declines to persist this session's transcript JSONL (only
    # the out-of-band ai-title lands). So --stream-session-events has nothing to
    # tail. The same flag therefore tells us, reliably and from the first
    # instant, that we must fall back to the live terminal scrape to give the
    # consumer any output. A normal top-level run never has it, so it never
    # falls back and never double-emits.
    is_child_session = bool(os.environ.get("CLAUDE_CODE_CHILD_SESSION"))
    # --stream-session-events: tail Claude Code's session JSONL and echo
    # each new canonical assistant/user event live, for full downstream
    # logging (tool calls + results, not just terminal text deltas).
    stream_session_events = getattr(args, "stream_session_events", False)
    session_tail: dict = {}

    # Mirror the PTY to --raw-log incrementally (flushed per chunk) so a
    # `tail -f` on it shows the live interactive screen. Without this the
    # transcript is only written once the session ends, which is useless
    # for diagnosing a run that's still in progress or wedged on a prompt.
    raw_fh = None
    if args.raw_log:
        try:
            p = Path(args.raw_log)
            p.parent.mkdir(parents=True, exist_ok=True)
            raw_fh = open(p, "wb", buffering=0)
        except OSError:
            raw_fh = None

    try:
        while time.time() - start < args.timeout_sec:
            now = time.time()
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                raw.extend(data)
                if raw_fh is not None:
                    try:
                        raw_fh.write(data)
                    except OSError:
                        pass
                last_output = time.time()

                if args.emit_terminal_delta:
                    emit(
                        {
                            "type": "tui_terminal_delta",
                            "text": clean_terminal(data.decode("utf-8", "replace")),
                            "uuid": str(uuid.uuid4()),
                            "session_id": args.session_id,
                        },
                        enabled=stream_json,
                    )

                snapshot = extract_assistant_snapshot(raw.decode("utf-8", "replace"))
                if snapshot and snapshot != last_snapshot:
                    # --stream-session-events supersedes terminal-scraped text
                    # deltas with real session events (below), so normally don't
                    # also emit the lower-fidelity scrape. BUT a child/subagent
                    # session's transcript is never persisted (see
                    # is_child_session above), so the tail emits nothing and the
                    # downstream log stays silent even though the agent is
                    # plainly working. In that case fall back to the live
                    # terminal scrape — the same surface the interactive screen
                    # renders. The emitted guard means that if a real session
                    # event ever does land we stop scraping and never double-emit.
                    session_events_silent = (
                        stream_session_events
                        and is_child_session
                        and session_tail.get("emitted", 0) == 0
                    )
                    delta = snapshot[len(last_snapshot) :] if snapshot.startswith(last_snapshot) else snapshot
                    if delta.strip():
                        if session_events_silent:
                            # Surface the scraped terminal text as a canonical
                            # `assistant` event (not a stream_event delta): a
                            # consumer that drives claude-p with
                            # --stream-session-events parses the session-event
                            # shape (assistant/user/result) and would drop a
                            # stream_event, so the live_tui_deltas shape used
                            # below wouldn't reach it. Emitting the incremental
                            # delta as an assistant text block gives that
                            # consumer live subagent output in the shape it
                            # already understands.
                            emit(
                                {
                                    "type": "assistant",
                                    "message": {
                                        "role": "assistant",
                                        "content": [{"type": "text", "text": delta}],
                                    },
                                    "session_id": args.session_id,
                                    "uuid": str(uuid.uuid4()),
                                },
                                enabled=stream_json,
                            )
                        elif args.live_tui_deltas and not stream_session_events:
                            emit(
                                {
                                    "type": "stream_event",
                                    "event": {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {"type": "text_delta", "text": delta},
                                    },
                                    "session_id": args.session_id,
                                    "parent_tool_use_id": None,
                                    "uuid": str(uuid.uuid4()),
                                },
                                enabled=stream_json,
                            )
                    last_snapshot = snapshot

            transcript = raw.decode("utf-8", "replace")
            block_type = classify_interactive_block(transcript)
            if block_type:
                if block_type == "workspace_trust_blocked" and args.trust_workspace and not trust_sent:
                    os.write(master, b"1\n")
                    trust_sent = True
                    last_output = time.time()
                    continue

                if block_type == "workspace_trust_blocked" and trust_sent:
                    pass
                else:
                    timed_out = False
                    break

            # The terminal surface is not a stable completion signal across
            # Claude Code versions and terminal modes. Poll the canonical
            # session JSONL while the TUI is running, and finish only after the
            # current session has a terminal assistant message. Tool-use turns
            # can also contain text and must not be mistaken for final output.
            if now - last_jsonl_poll >= 0.5:
                last_jsonl_poll = now
                if stream_session_events:
                    _drain_session_events(args.session_id, session_tail, stream_json)
                persisted = read_persisted_assistant(args.session_id)
                if persisted and persisted.get("text") and persisted.get("terminal"):
                    timed_out = False
                    break

            if last_snapshot and time.time() - last_output >= args.quiet_after_sec:
                persisted = read_persisted_assistant(args.session_id)
                if not persisted or persisted.get("terminal"):
                    timed_out = False
                    break
            if proc.poll() is not None:
                timed_out = False
                break
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        os.close(master)
        if raw_fh is not None:
            try:
                raw_fh.close()
            except OSError:
                pass

    # Final drain: claude persists the terminal assistant turn before the
    # loop's break condition fires, but flush any remaining new events once
    # more so the last assistant/tool_result event is guaranteed emitted.
    if stream_session_events:
        _drain_session_events(args.session_id, session_tail, stream_json)

    transcript = clean_terminal(raw.decode("utf-8", "replace"))
    answer = extract_assistant_snapshot(transcript)
    return transcript, answer, proc.returncode, timed_out, start


def doctor(args: argparse.Namespace) -> int:
    """Print diagnostics that explain most installation and local CLI failures."""
    print("claude-p doctor")
    print(f"invoked_as: {sys.argv[0]}")
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"cwd: {args.cwd}")
    print(f"home: {Path.home()}")

    claude_path = shutil.which("claude")
    print(f"claude_path: {claude_path or 'not found'}")
    if claude_path:
        try:
            proc = subprocess.run(
                ["claude", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = (proc.stdout or proc.stderr).strip()
            print(f"claude_version: {version or 'unknown'}")
        except Exception as exc:  # pragma: no cover - defensive diagnostic path.
            print(f"claude_version_error: {exc}")

    session_root = _claude_config_dir() / "projects"
    print(f"session_root: {session_root}")
    print(f"session_root_exists: {session_root.exists()}")
    print(f"session_root_writable: {os.access(session_root, os.W_OK) if session_root.exists() else False}")

    claude_p_path = shutil.which("claude-p")
    print(f"claude_p_path: {claude_p_path or 'not found'}")
    present_overrides = [name for name in SUBSCRIPTION_BACKEND_ENV_OVERRIDES if os.environ.get(name)]
    print(f"provider_env_overrides_present: {','.join(present_overrides) if present_overrides else 'none'}")
    print(f"provider_env_policy: {'preserve' if args.preserve_provider_env else 'strip_for_subscription_backend'}")
    print("smoke_test:")
    print('  claude-p "Respond exactly: CLAUDE_P_OK" --timeout-sec 45 --quiet-after-sec 2 --raw-log /tmp/claude-p-smoke.raw.log')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--cwd", default=os.getcwd())

    # Common Claude Code options. The goal is CLI compatibility with the print
    # path while still using the interactive TUI backend internally.
    parser.add_argument("-p", "--print", dest="print_mode", action="store_true", help="Accepted for claude -p compatibility.")
    parser.add_argument("--add-dir", nargs="+", action="append", default=[])
    parser.add_argument("--agent")
    parser.add_argument("--agents")
    parser.add_argument("--allow-dangerously-skip-permissions", action="store_true")
    parser.add_argument("--allowedTools", "--allowed-tools", dest="allowed_tools", nargs="+", action="append", default=[])
    parser.add_argument("--append-system-prompt")
    parser.add_argument("--bare", action="store_true")
    parser.add_argument("--betas", nargs="+", action="append", default=[])
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--chrome", action="store_true")
    parser.add_argument("--no-chrome", action="store_true")
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true")
    parser.add_argument("--dangerously-skip-permissions", action="store_true")
    parser.add_argument("-d", "--debug", nargs="?", const="")
    parser.add_argument("--debug-file")
    parser.add_argument("--disable-slash-commands", action="store_true")
    parser.add_argument("--disallowedTools", "--disallowed-tools", dest="disallowed_tools", nargs="+", action="append", default=[])
    parser.add_argument("--effort")
    parser.add_argument("--exclude-dynamic-system-prompt-sections", action="store_true")
    parser.add_argument("--fallback-model")
    parser.add_argument("--file", dest="files", nargs="+", action="append", default=[])
    parser.add_argument("--fork-session", action="store_true")
    parser.add_argument("--from-pr", nargs="?", const="")
    parser.add_argument("--ide", action="store_true")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--tools", nargs="+", default=["default"])
    parser.add_argument("--permission-mode", default="default")
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Output format, matching claude -p. Default: text.",
    )
    parser.add_argument("--verbose", action="store_true", help="Accepted for claude -p CLI compatibility.")
    parser.add_argument("--include-hook-events", action="store_true")
    parser.add_argument(
        "--include-partial-messages",
        action="store_true",
        help="Accepted for claude -p CLI compatibility. With the TUI backend, stream-json emits one final text delta by default.",
    )
    parser.add_argument("--input-format", choices=["text", "stream-json"], default="text")
    parser.add_argument("--json-schema")
    parser.add_argument("--max-budget-usd")
    parser.add_argument("--mcp-config", nargs="+", action="append", default=[])
    parser.add_argument("--mcp-debug", action="store_true")
    parser.add_argument("-n", "--name")
    parser.add_argument("--no-session-persistence", action="store_true")
    parser.add_argument("--plugin-dir", action="append", default=[])
    parser.add_argument("--plugin-url", action="append", default=[])
    parser.add_argument("--remote-control", nargs="?", const="")
    parser.add_argument("--remote-control-session-name-prefix")
    parser.add_argument("--replay-user-messages", action="store_true")
    parser.add_argument("-r", "--resume", nargs="?", const="")
    parser.add_argument("--setting-sources")
    parser.add_argument("--settings")
    parser.add_argument("--strict-mcp-config", action="store_true")
    parser.add_argument("--system-prompt")
    parser.add_argument("--tmux", nargs="?", const="")
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("-w", "--worktree", nargs="?", const="")

    # Wrapper-only controls.
    parser.add_argument("--timeout-sec", type=float, default=90)
    parser.add_argument("--quiet-after-sec", type=float, default=3)
    parser.add_argument("--session-id", default=str(uuid.uuid4()))
    parser.add_argument("--term", default="xterm-256color")
    parser.add_argument("--raw-log")
    parser.add_argument(
        "--preserve-provider-env",
        action="store_true",
        help="Preserve Anthropic API/provider environment variables instead of stripping them for the subscription-backed TUI.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Print local installation diagnostics without calling the model.",
    )
    parser.add_argument("--emit-terminal-delta", action="store_true")
    parser.add_argument(
        "--live-tui-deltas",
        action="store_true",
        help="Emit live text deltas from the lossy TUI surface. Default buffers until persisted JSONL final text is available.",
    )
    parser.add_argument(
        "--stream-session-events",
        action="store_true",
        help=(
            "Tail Claude Code's session JSONL and echo each new canonical "
            "assistant/user event (text, tool_use, tool_result) live, in the "
            "same stream-json shape as `claude -p`. Gives a downstream "
            "consumer full live logging (tool calls + results), not just "
            "scraped terminal text. Supersedes --live-tui-deltas when set."
        ),
    )
    parser.add_argument(
        "--trust-workspace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically trust the workspace if prompted by Claude Code (default: true).",
    )
    parser.add_argument(
        "--isolate-config-dir",
        action="store_true",
        help=(
            "Run this claude in a throwaway CLAUDE_CONFIG_DIR that symlinks the "
            "base dir's auth/plugins/caches but keeps a fresh session store. "
            "Lets several claude-p TUIs run concurrently (e.g. a dispatcher + "
            "its subagents) without the shared-config-dir clash that otherwise "
            "stops the non-first session from persisting (leaving "
            "--stream-session-events empty)."
        ),
    )
    args = parser.parse_args()
    recover_prompt_from_variadic_args(args)

    # Must run before anything reads the config dir (session discovery, env).
    if getattr(args, "isolate_config_dir", False):
        _isolate_config_dir()

    if args.doctor:
        return doctor(args)

    if args.version:
        subprocess.run(["claude", "--version"], check=False)
        return 0

    if args.prompt is None:
        if sys.stdin.isatty():
            parser.error("prompt is required unless stdin provides input")
        args.prompt = sys.stdin.read()

    unsupported: list[str] = []
    if args.input_format != "text":
        unsupported.append("--input-format stream-json")
    if args.replay_user_messages:
        unsupported.append("--replay-user-messages")
    if args.no_session_persistence:
        unsupported.append("--no-session-persistence")
    if args.bare:
        unsupported.append("--bare")
    if args.max_budget_usd:
        unsupported.append("--max-budget-usd")
    if args.fallback_model:
        unsupported.append("--fallback-model")
    if unsupported:
        for flag in unsupported:
            warn(f"{flag} is not supported by the interactive subscription backend; continuing without exact claude -p semantics")

    stream_json = args.output_format == "stream-json"
    message_id = f"msg_tui_{uuid.uuid4().hex[:24]}"
    start = time.time()

    emit(
        {
            "type": "system",
            "subtype": "init",
            "cwd": args.cwd,
            "session_id": args.session_id,
            "tools": [],
            "mcp_servers": [],
            "model": args.model,
            "permissionMode": args.permission_mode,
            "apiKeySource": "interactive_tui_subscription",
            "claude_code_version": None,
            "output_style": "default",
            "uuid": str(uuid.uuid4()),
            "fast_mode_state": "off",
        },
        enabled=stream_json,
    )
    emit(
        {
            "type": "system",
            "subtype": "status",
            "status": "requesting",
            "uuid": str(uuid.uuid4()),
            "session_id": args.session_id,
        },
        enabled=stream_json,
    )
    emit(
        {
            "type": "stream_event",
            "event": {
                "type": "message_start",
                "message": {
                    "model": args.model,
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "stop_details": None,
                    "usage": {
                        "input_tokens": None,
                        "cache_creation_input_tokens": None,
                        "cache_read_input_tokens": None,
                        "output_tokens": None,
                        "service_tier": None,
                    },
                },
            },
            "session_id": args.session_id,
            "parent_tool_use_id": None,
            "uuid": str(uuid.uuid4()),
            "ttft_ms": None,
        },
        enabled=stream_json,
    )
    emit(
        {
            "type": "stream_event",
            "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            "session_id": args.session_id,
            "parent_tool_use_id": None,
            "uuid": str(uuid.uuid4()),
        },
        enabled=stream_json,
    )

    transcript, tui_answer, exit_code, timed_out, run_start = run_tui(args, stream_json)
    # run_tui mirrors the PTY to --raw-log live; no end-of-run write needed.

    all_assistant_events = read_all_assistant_events(args.session_id)
    persisted = read_persisted_assistant(args.session_id, require_terminal=True)
    answer = persisted["text"] if persisted else tui_answer
    final_answer_source = "session_jsonl" if persisted else "tui_transcript"
    if persisted and tui_answer and tui_answer != persisted["text"]:
        canonical = canonical_json_if_equivalent(tui_answer, persisted["text"])
        if canonical is not None:
            answer = canonical
            final_answer_source = "json_canonicalized_from_matching_tui_and_session_jsonl"
    final_model = persisted.get("model") if persisted else args.model
    message_id = persisted.get("message_id") if persisted and persisted.get("message_id") else message_id

    # num_turns: each assistant event is one turn; tool-use rounds have
    # stop_reason "tool_use", the final response has "end_turn".
    num_turns = len(all_assistant_events) if all_assistant_events else 1

    # usage: real token data from JSONL, falling back to word-count estimate.
    usage = build_usage_from_events(all_assistant_events) if all_assistant_events else build_usage(answer)

    if answer and not args.live_tui_deltas and not args.stream_session_events:
        emit(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": answer},
                },
                "session_id": args.session_id,
                "parent_tool_use_id": None,
                "uuid": str(uuid.uuid4()),
            },
            enabled=stream_json,
        )

    failure = classify_failure(transcript, answer, timed_out)
    is_error = failure is not None
    duration_ms = now_ms(start)

    if args.output_format == "text":
        if is_error:
            if answer:
                print(answer, file=sys.stderr)
            print(
                f"claude-p error: {failure}. "
                "Run with --raw-log /tmp/claude-p.raw.log and inspect the log if this is unexpected.",
                file=sys.stderr,
            )
        elif answer:
            print(answer)
        return 0 if not is_error else 2

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success" if not is_error else "error",
                    "is_error": is_error,
                    "duration_ms": duration_ms,
                    "duration_api_ms": None,
                    "num_turns": num_turns,
                    "result": answer,
                    "session_id": args.session_id,
                    "total_cost_usd": None,
                    "usage": usage,
                    "terminal_reason": "completed" if not is_error else failure,
                    "interactive_tui_backend": {
                        "raw_log": args.raw_log,
                        "session_jsonl": persisted.get("path") if persisted else None,
                        "tui_answer": tui_answer,
                        "final_answer_source": final_answer_source,
                        "timed_out": timed_out,
                        "exit_code": exit_code,
                        "extraction_confidence": "high" if persisted else ("medium" if answer else "none"),
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return 0 if not is_error else 2

    # When streaming real session events, the terminal assistant turn was
    # already emitted live from the session JSONL — don't re-emit a
    # synthetic copy (it would duplicate the final text downstream).
    if not args.stream_session_events:
      emit(
        {
            "type": "assistant",
            "message": {
                "model": final_model,
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": answer}] if answer else [],
                "stop_reason": "end_turn" if not is_error else None,
                "stop_sequence": None,
                "stop_details": None,
                "usage": {
                    "input_tokens": None,
                    "cache_creation_input_tokens": None,
                    "cache_read_input_tokens": None,
                    "output_tokens": usage["output_tokens"],
                    "service_tier": None,
                },
                "context_management": None,
            },
            "parent_tool_use_id": None,
            "session_id": args.session_id,
            "uuid": str(uuid.uuid4()),
        }
    )
    emit(
        {
            "type": "stream_event",
            "event": {"type": "content_block_stop", "index": 0},
            "session_id": args.session_id,
            "parent_tool_use_id": None,
            "uuid": str(uuid.uuid4()),
        }
    )
    emit(
        {
            "type": "stream_event",
            "event": {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn" if not is_error else "error", "stop_sequence": None, "stop_details": None},
                "usage": {
                    "input_tokens": None,
                    "cache_creation_input_tokens": None,
                    "cache_read_input_tokens": None,
                    "output_tokens": usage["output_tokens"],
                    "iterations": usage["iterations"],
                },
                "context_management": {"applied_edits": []},
            },
            "session_id": args.session_id,
            "parent_tool_use_id": None,
            "uuid": str(uuid.uuid4()),
        }
    )
    emit(
        {
            "type": "stream_event",
            "event": {"type": "message_stop"},
            "session_id": args.session_id,
            "parent_tool_use_id": None,
            "uuid": str(uuid.uuid4()),
        }
    )
    emit(
        {
            "type": "rate_limit_event",
            "rate_limit_info": {"status": "unknown"},
            "session_id": args.session_id,
            "uuid": str(uuid.uuid4()),
        }
    )
    emit(
        {
            "type": "result",
            "subtype": "success" if not is_error else "error",
            "is_error": is_error,
            "api_error_status": None,
            "duration_ms": duration_ms,
            "duration_api_ms": None,
            "num_turns": num_turns,
            "result": answer,
            "stop_reason": "end_turn" if not is_error else None,
            "session_id": args.session_id,
            "total_cost_usd": None,
            "usage": usage,
            "modelUsage": {},
            "permission_denials": [],
            "terminal_reason": "completed" if not is_error else failure,
            "fast_mode_state": "off",
            "uuid": str(uuid.uuid4()),
            "interactive_tui_backend": {
                "raw_log": args.raw_log,
                "session_jsonl": persisted.get("path") if persisted else None,
                "tui_answer": tui_answer,
                "final_answer_source": final_answer_source,
                "timed_out": timed_out,
                "exit_code": exit_code,
                "extraction_confidence": "high" if persisted else ("medium" if answer else "none"),
                "compatibility_note": "Shape-compatible with claude -p stream-json core events; usage/cost/tool events are best-effort because TUI has no machine protocol.",
            },
        }
    )

    return 0 if not is_error else 2


if __name__ == "__main__":
    raise SystemExit(main())
