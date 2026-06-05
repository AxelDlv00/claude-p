"""Tests for `--stream-session-events`: tailing Claude Code's session JSONL
and re-emitting canonical assistant/user events as stream-json.

These let a downstream stream-json consumer show full live logging (text,
tool calls, tool results) instead of only terminal-scraped text deltas.
"""

import json

from claude_p.cli import _drain_session_events


def _write(path, events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def _emitted(capsys):
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_drain_emits_assistant_and_tool_result_user(tmp_path, capsys):
    session = tmp_path / "sid.jsonl"
    _write(session, [
        # The initial human prompt is a string-content user event — must be
        # skipped (not useful downstream, and would break block-iterating
        # consumers).
        {"type": "user", "message": {"role": "user", "content": "prove it"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "running build"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "lake build"}},
        ]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "ok"},
        ]}},
    ])
    state = {"path": session, "offset": 0}
    _drain_session_events("sid", state, True)
    rows = _emitted(capsys)
    types = [r["type"] for r in rows]
    # assistant echoed; only the tool-result user echoed (prompt skipped).
    assert types == ["assistant", "user"]
    assert rows[1]["message"]["content"][0]["type"] == "tool_result"


def test_drain_is_incremental_via_offset(tmp_path, capsys):
    session = tmp_path / "sid.jsonl"
    _write(session, [
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": [{"type": "text", "text": "first"}]}},
    ])
    state = {"path": session, "offset": 0}
    _drain_session_events("sid", state, True)
    assert len(_emitted(capsys)) == 1

    # Append a second turn; a re-drain must emit only the new event.
    with session.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "assistant", "message": {"role": "assistant",
                "content": [{"type": "text", "text": "second"}]}}) + "\n")
    _drain_session_events("sid", state, True)
    rows = _emitted(capsys)
    assert len(rows) == 1
    assert rows[0]["message"]["content"][0]["text"] == "second"


def test_drain_holds_back_partial_trailing_line(tmp_path, capsys):
    session = tmp_path / "sid.jsonl"
    # A complete line followed by a partial (no trailing newline).
    full = json.dumps({"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "complete"}]}})
    session.write_text(full + "\n" + '{"type":"assist', encoding="utf-8")
    state = {"path": session, "offset": 0}
    _drain_session_events("sid", state, True)
    rows = _emitted(capsys)
    assert len(rows) == 1
    assert rows[0]["message"]["content"][0]["text"] == "complete"


def test_drain_missing_file_is_noop(tmp_path, capsys):
    # No file located yet (path stays None) → nothing emitted, no error.
    _drain_session_events("does-not-exist", {}, True)
    assert _emitted(capsys) == []
