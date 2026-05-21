from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Any, AsyncIterator, Iterable

from .types import AssistantMessage, ResultMessage, SDKMessage, StreamEventMessage, SystemMessage


@dataclass
class ClaudePOptions:
    """Options for the interactive Claude Code `claude -p` fallback.

    The names mirror the official Claude Agent SDK's options style while
    preserving CLI compatibility with `claude -p`.
    """

    cwd: str | None = None
    model: str = "sonnet"
    tools: str | Iterable[str] = "default"
    permission_mode: str = "default"
    output_format: str = "stream-json"
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    mcp_config: list[str] = field(default_factory=list)
    settings: str | None = None
    session_id: str | None = None
    timeout_sec: float = 90
    quiet_after_sec: float = 3
    raw_log: str | None = None
    include_partial_messages: bool = True
    trust_workspace: bool = False
    executable: str | None = None
    extra_args: list[str] = field(default_factory=list)

    def command(self, prompt: str) -> list[str]:
        if self.executable:
            cmd = [self.executable, prompt]
        else:
            cmd = [sys.executable, "-m", "claude_p.cli", prompt]
        cmd.extend(["--model", self.model])
        if isinstance(self.tools, str):
            cmd.extend(["--tools", self.tools])
        else:
            cmd.extend(["--tools", ",".join(self.tools)])
        cmd.extend(["--permission-mode", self.permission_mode])
        cmd.extend(["--output-format", self.output_format])
        cmd.extend(["--timeout-sec", str(self.timeout_sec)])
        cmd.extend(["--quiet-after-sec", str(self.quiet_after_sec)])
        if self.cwd:
            cmd.extend(["--cwd", self.cwd])
        if self.system_prompt:
            cmd.extend(["--system-prompt", self.system_prompt])
        if self.append_system_prompt:
            cmd.extend(["--append-system-prompt", self.append_system_prompt])
        for tool in self.allowed_tools:
            cmd.extend(["--allowedTools", tool])
        for tool in self.disallowed_tools:
            cmd.extend(["--disallowedTools", tool])
        for config in self.mcp_config:
            cmd.extend(["--mcp-config", config])
        if self.settings:
            cmd.extend(["--settings", self.settings])
        if self.session_id:
            cmd.extend(["--session-id", self.session_id])
        if self.raw_log:
            cmd.extend(["--raw-log", self.raw_log])
        if self.include_partial_messages:
            cmd.append("--include-partial-messages")
        if self.trust_workspace:
            cmd.append("--trust-workspace")
        cmd.extend(self.extra_args)
        return cmd


def _message_from_raw(raw: dict[str, Any]) -> SDKMessage:
    typ = raw.get("type")
    if typ == "system":
        return SystemMessage(type="system", raw=raw, subtype=raw.get("subtype"))
    if typ == "stream_event":
        return StreamEventMessage(type="stream_event", raw=raw, event=raw.get("event"))
    if typ == "assistant":
        content = raw.get("message", {}).get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return AssistantMessage(type="assistant", raw=raw, text="".join(text_parts))
    if typ == "result":
        return ResultMessage(
            type="result",
            raw=raw,
            result=raw.get("result", ""),
            is_error=bool(raw.get("is_error")),
            session_id=raw.get("session_id"),
            terminal_reason=raw.get("terminal_reason"),
        )
    return SDKMessage(type=str(typ or "raw"), raw=raw)


async def query(prompt: str, *, options: ClaudePOptions | None = None) -> AsyncIterator[SDKMessage]:
    """Run a prompt and yield stream-json messages.

    This mirrors the official Claude Agent SDK's `query(...)` shape, but the
    backend is interactive Claude Code instead of `claude -p`.
    """

    options = options or ClaudePOptions()
    options.output_format = "stream-json"
    proc = await asyncio.create_subprocess_exec(
        *options.command(prompt),
        cwd=options.cwd or os.getcwd(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    async for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line:
            continue
        yield _message_from_raw(json.loads(line))
    stderr = await proc.stderr.read() if proc.stderr else b""
    code = await proc.wait()
    if code != 0:
        raise RuntimeError(stderr.decode("utf-8", "replace") or f"claude-p exited with {code}")


class ClaudePClient:
    """Small client wrapper inspired by ClaudeSDKClient."""

    def __init__(self, options: ClaudePOptions | None = None):
        self.options = options or ClaudePOptions()

    async def __aenter__(self) -> "ClaudePClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def query(self, prompt: str) -> AsyncIterator[SDKMessage]:
        async for message in query(prompt, options=self.options):
            yield message

    async def run(self, prompt: str) -> ResultMessage:
        result: ResultMessage | None = None
        async for message in self.query(prompt):
            if isinstance(message, ResultMessage):
                result = message
        if result is None:
            raise RuntimeError("claude-p did not emit a result message")
        return result
