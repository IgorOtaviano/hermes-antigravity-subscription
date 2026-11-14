"""Multipart delivery: a long input reaches agy whole, as several messages.

agy cuts a single user message at a fixed size (183,000 to 194,000 UTF-8 bytes,
measured on agy 1.5.9) and drops the rest silently, while a conversation keeps
everything it was told. The worker path therefore splits a long prompt into
consecutive messages instead of shortening it. These tests pin the split, the
frame, the client's delivery loop and the budgets that no longer apply.
"""

from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import prompt as agy_prompt
from client import AntigravityClient


class SplitTests(unittest.TestCase):
    def test_parts_join_back_to_the_text_and_respect_the_limit(self):
        text = "\n\n".join(f"block {i} " + "x" * 700 for i in range(40))
        parts = agy_prompt.split_prompt_parts(text, 3000)
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), text)
        for part in parts:
            self.assertLessEqual(len(part.encode("utf-8")), 3000)

    def test_cuts_fall_on_blank_lines_when_blocks_fit(self):
        blocks = [f"block {i} " + "x" * 700 for i in range(10)]
        parts = agy_prompt.split_prompt_parts("\n\n".join(blocks), 3000)
        for part in parts[:-1]:
            self.assertTrue(part.endswith("\n\n"), "a part ended inside a block")

    def test_multibyte_text_is_measured_in_bytes_and_never_cut_inside_a_character(self):
        text = "\n".join("revisão técnica não é só ação " * 20 for _ in range(30))
        parts = agy_prompt.split_prompt_parts(text, 2000)
        self.assertEqual("".join(parts), text)  # decoding each piece worked
        for part in parts:
            self.assertLessEqual(len(part.encode("utf-8")), 2000)
        # More bytes than characters: a character count would overshoot.
        self.assertGreater(len(text.encode("utf-8")), len(text))

    def test_a_single_line_larger_than_a_part_is_cut_inside(self):
        text = "é" * 5000  # 10,000 bytes, no line end anywhere
        parts = agy_prompt.split_prompt_parts(text, 1500)
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(len(p.encode("utf-8")) <= 1500 for p in parts))

    def test_the_last_part_is_filled_from_the_end(self):
        blocks = [f"block {i} " + "x" * 700 for i in range(9)] + ["THE REQUEST", "its instruction " + "i" * 300]
        parts = agy_prompt.split_prompt_parts("\n\n".join(blocks), 3000)
        self.assertIn("THE REQUEST", parts[-1])
        self.assertIn("block 8 ", parts[-1])  # not a leftover: as much as fits
        self.assertGreater(len(parts[-1].encode("utf-8")), 1500)

    def test_text_that_fits_is_one_part(self):
        self.assertEqual(agy_prompt.split_prompt_parts("short", 3000), ["short"])
        self.assertEqual(agy_prompt.split_prompt_parts("x" * 9000, 0), ["x" * 9000])

    def test_the_frame_fits_its_reserve_and_names_the_last_part(self):
        middle = agy_prompt.frame_prompt_part("BODY", 998, 999)
        last = agy_prompt.frame_prompt_part("BODY", 999, 999)
        for framed in (middle, last):
            self.assertLessEqual(
                len(framed.encode("utf-8")) - len("BODY"), agy_prompt._PART_FRAME_BYTES
            )
        self.assertIn(f"Reply with exactly: {agy_prompt.MULTIPART_ACK}", middle)
        self.assertIn("FINAL PART", last)
        self.assertNotIn("Reply with exactly", last)
        self.assertEqual(agy_prompt.frame_prompt_part("BODY", 1, 1), "BODY")


class SettingsTests(unittest.TestCase):
    def test_a_setting_that_is_not_an_integer_keeps_the_default(self):
        with patch.dict("os.environ", {"HERMES_AGY_PROMPT_MAX_CHARS": "120k"}):
            with self.assertLogs("prompt", level="WARNING"):
                self.assertEqual(agy_prompt._env_int("HERMES_AGY_PROMPT_MAX_CHARS", 120_000), 120_000)
        with patch.dict("os.environ", {"HERMES_AGY_PROMPT_MAX_CHARS": " 5000 "}):
            self.assertEqual(agy_prompt._env_int("HERMES_AGY_PROMPT_MAX_CHARS", 120_000), 5000)

    def test_multipart_is_off_at_zero(self):
        with patch.object(agy_prompt, "_PART_MAX_BYTES", 0):
            self.assertFalse(agy_prompt.multipart_enabled())
        self.assertTrue(agy_prompt.multipart_enabled())


def _tool_loop(results: list[str], request: str = "review the diff") -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": request}]
    for i, result in enumerate(results):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"call_{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": result})
    return messages


class BudgetTests(unittest.TestCase):
    """What used to be shortened to fit one message, with multipart on and off."""

    def test_a_large_tool_result_is_whole_in_a_multipart_delta(self):
        result = "HEAD " + "d" * 100_000 + " MIDDLE-MARK " + "d" * 100_000 + " TAIL"
        delta = _tool_loop([result])[1:]
        whole = agy_prompt._format_delta_prompt(delta, pending_user_text="review", multipart=True)
        self.assertIn("MIDDLE-MARK", whole)
        self.assertNotIn("large tool output truncated", whole)
        legacy = agy_prompt._format_delta_prompt(delta, pending_user_text="review")
        self.assertNotIn("MIDDLE-MARK", legacy)
        self.assertIn("large tool output truncated", legacy)

    def test_older_tool_results_are_kept_whole_while_the_history_fits(self):
        results = [f"RESULT-{i} " + "r" * 30_000 + f" END-{i}" for i in range(12)]
        messages = _tool_loop(results)
        whole = agy_prompt._format_messages_as_prompt(messages, multipart=True)
        for i in range(12):
            self.assertIn(f"END-{i}", whole)
        self.assertNotIn("tool output truncated", whole)
        # One message: only the last 8 stay whole, each cut at 20,000.
        legacy = agy_prompt._format_messages_as_prompt(messages)
        self.assertNotIn("END-0 ", legacy + " ")
        self.assertIn("tool output truncated", legacy)

    def test_older_tool_results_are_pruned_when_the_history_does_not_fit(self):
        results = [f"RESULT-{i} " + "r" * 30_000 + f" END-{i}" for i in range(12)]
        with patch.object(agy_prompt, "_WORKER_PROMPT_MAX_CHARS", 200_000):
            pruned = agy_prompt._format_messages_as_prompt(_tool_loop(results), multipart=True)
        self.assertIn("tool output truncated", pruned)
        self.assertIn("END-11", pruned)
        self.assertLessEqual(len(pruned), 200_000)

    def test_tool_schemas_are_never_cut_by_the_budget(self):
        schemas = "Available tools (OpenAI function schema):\n" + json.dumps(
            [{"type": "function", "function": {"name": f"t{i}", "description": "d" * 900}} for i in range(60)]
        )
        sections = [
            "preamble",
            schemas,
            "### SYSTEM INSTRUCTIONS (HERMES AGENT):\nSystem:\nrules",
            "Conversation transcript:\n\n" + "\n\n".join(f"User:\nmessage {i} " + "m" * 500 for i in range(80)),
            "### LATEST USER REQUEST TO ANSWER:\nUser:\nnow",
        ]
        fitted = agy_prompt._fit_prompt_sections(sections, limit=30_000)
        self.assertIn(schemas, fitted)
        json.loads(schemas.split("\n", 1)[1])  # still what was written

    def test_a_long_request_is_not_repeated_whole_on_every_tool_turn(self):
        request = "REQUEST-HEAD " + "q" * 30_000 + " REQUEST-TAIL"
        messages = _tool_loop(["small result"], request=request)
        delta = agy_prompt._format_delta_prompt(messages[1:], pending_user_text=request, multipart=True)
        self.assertIn("REQUEST-HEAD", delta)
        self.assertIn("REQUEST-TAIL", delta)
        self.assertIn("not repeated here", delta)
        self.assertLess(len(delta), 8_000)
        # The section that delivers the request itself keeps every character.
        first = agy_prompt._format_messages_as_prompt([{"role": "user", "content": request}], multipart=True)
        self.assertIn(request, first)

    def test_a_short_request_is_restated_unchanged(self):
        self.assertEqual(agy_prompt._bounded_restatement("fix the login bug"), "fix the login bug")


def _result(text: str, input_tokens: int, conversation_id: str = "conv-1") -> str:
    return "\n".join(
        [
            json.dumps({"event": "init", "conversation_id": conversation_id}),
            json.dumps({"event": "step_update", "step_update": {"text_delta": text}}),
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "status": "SUCCESS",
                        "response": text + "\n",
                        "usage": {"input_tokens": input_tokens, "output_tokens": 1, "total_tokens": input_tokens + 1},
                    },
                }
            ),
        ]
    ) + "\n"


def _worker(stdout_text: str) -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = io.StringIO(stdout_text)
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None  # a live worker
    proc.wait.return_value = 0
    return proc


def _written(proc: MagicMock) -> list[str]:
    """The content of every user event written to the worker, in order."""
    contents = []
    for call in proc.stdin.write.call_args_list:
        event = json.loads(call.args[0])
        if event.get("event") == "user":
            contents.append(event["message"]["content"])
    return contents


class ClientDeliveryTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch("client.is_authenticated", return_value=True),
            patch("client.resolve_real_token_path", return_value=None),
            patch("client.resolve_agy_command", return_value="agy"),
            patch("process._link_macos_keychains"),
            # Small parts, so a short prompt is already a long one.
            patch.object(agy_prompt, "_PART_MAX_BYTES", 3000),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        import models

        models._catalog_cache = (time.monotonic() + 3600, {})
        self.addCleanup(lambda: setattr(models, "_catalog_cache", None))
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.client = AntigravityClient(cwd=tmp_dir.name)
        self.addCleanup(self.client.close)
        patcher = patch.object(self.client, "_terminate_process")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _long_messages(self) -> list[dict]:
        messages: list[dict] = []
        for i in range(6):
            messages.append({"role": "user", "content": f"QUESTION-{i} " + "u" * 600})
            messages.append({"role": "assistant", "content": f"ANSWER-{i} " + "a" * 600})
        messages.append({"role": "user", "content": "FINAL-REQUEST"})
        return messages

    def test_a_long_full_send_is_delivered_in_parts_and_nothing_is_lost(self):
        client = self.client
        messages = self._long_messages()
        payload = agy_prompt._format_messages_as_prompt(messages, multipart=True)
        expected_parts = agy_prompt.split_prompt_parts(payload)
        self.assertGreater(len(expected_parts), 2)
        acks = "".join(_result("ACK", 100 * (i + 1)) for i in range(len(expected_parts) - 1))
        proc = _worker(acks + _result("the answer", 100 * len(expected_parts)))

        with patch("subprocess.Popen", return_value=proc) as popen:
            res = client.chat.completions.create(
                model="gemini-3.8-flash-high", messages=messages, stream=False
            )

        self.assertEqual(res.choices[0].message.content, "the answer")
        self.assertEqual(popen.call_count, 1)
        written = _written(proc)
        self.assertEqual(len(written), len(expected_parts))
        for content in written:
            self.assertLessEqual(len(content.encode("utf-8")), 3000)
        for content in written[:-1]:
            self.assertIn("Reply with exactly: ACK", content)
        self.assertIn("FINAL PART", written[-1])
        self.assertIn("FINAL-REQUEST", written[-1])
        everything = "\n".join(written)
        for i in range(6):
            self.assertIn(f"QUESTION-{i}", everything)
            self.assertIn(f"ANSWER-{i}", everything)
        # The tokens of every part are reported once, with the real turn.
        self.assertEqual(res.usage.prompt_tokens, 100 * len(expected_parts))
        # The acknowledgements never became conversation history.
        self.assertEqual(client._conversation_history, messages)
        self.assertFalse(client._worker_lock.locked())

    def test_the_next_turn_is_a_continuation_after_a_multipart_send(self):
        client = self.client
        messages = self._long_messages()
        parts = agy_prompt.split_prompt_parts(agy_prompt._format_messages_as_prompt(messages, multipart=True))
        stdout = "".join(_result("ACK", 10) for _ in parts[:-1]) + _result("first", 50) + _result("second", 60)
        proc = _worker(stdout)

        with patch("subprocess.Popen", return_value=proc) as popen:
            client.chat.completions.create(model="gemini-3.8-flash-high", messages=messages, stream=False)
            follow_up = [*messages, {"role": "assistant", "content": "first"}, {"role": "user", "content": "and then?"}]
            res = client.chat.completions.create(model="gemini-3.8-flash-high", messages=follow_up, stream=False)

        self.assertEqual(res.choices[0].message.content, "second")
        self.assertEqual(popen.call_count, 1)  # same worker, no full resend
        last = _written(proc)[-1]
        self.assertIn("and then?", last)
        self.assertNotIn("QUESTION-0", last)

    def test_a_long_tool_result_is_delivered_in_parts_on_a_continuation(self):
        client = self.client
        first = [{"role": "user", "content": "review the diff"}]
        result = "\n\n".join(f"HUNK-{i} " + "h" * 500 for i in range(12))
        second = [
            *first,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_diff", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": result},
        ]
        delta = agy_prompt._format_delta_prompt(second[1:], pending_user_text="review the diff", multipart=True)
        acks = len(agy_prompt.split_prompt_parts(delta)) - 1
        self.assertGreater(acks, 0)
        proc = _worker(
            _result("calling", 10) + "".join(_result("ACK", 20) for _ in range(acks)) + _result("reviewed", 90)
        )

        with patch("subprocess.Popen", return_value=proc):
            client.chat.completions.create(model="gemini-3.8-flash-high", messages=first, stream=False)
            before = len(_written(proc))
            chunks = list(
                client.chat.completions.create(model="gemini-3.8-flash-high", messages=second, stream=True)
            )

        delta_writes = _written(proc)[before:]
        self.assertGreater(len(delta_writes), 1)
        everything = "\n".join(delta_writes)
        for i in range(12):
            self.assertIn(f"HUNK-{i} ", everything)
        text = "".join(
            choice.delta.content or ""
            for chunk in chunks
            for choice in getattr(chunk, "choices", [])
            if getattr(choice, "delta", None) is not None
        )
        self.assertIn("reviewed", text)

    def test_a_part_that_gets_no_result_fails_the_turn(self):
        client = self.client
        messages = self._long_messages()
        proc = _worker(_result("ACK", 10))  # the second part's turn never answers
        proc.poll.side_effect = None

        with patch("subprocess.Popen", return_value=proc):
            with self.assertRaises(RuntimeError):
                client.chat.completions.create(
                    model="gemini-3.8-flash-high", messages=messages, stream=False, timeout=2
                )

        self.assertEqual(client._conversation_history, [])
        self.assertIsNone(client._worker_proc)
        self.assertFalse(client._worker_lock.locked())

    def test_multipart_off_writes_one_unframed_message(self):
        client = self.client
        messages = self._long_messages()
        proc = _worker(_result("the answer", 100))
        with patch.object(agy_prompt, "_PART_MAX_BYTES", 0), patch("subprocess.Popen", return_value=proc):
            res = client.chat.completions.create(
                model="gemini-3.8-flash-high", messages=messages, stream=False
            )
        self.assertEqual(res.choices[0].message.content, "the answer")
        written = _written(proc)
        self.assertEqual(len(written), 1)
        self.assertNotIn("MULTIPART INPUT", written[0])


if __name__ == "__main__":
    unittest.main()
