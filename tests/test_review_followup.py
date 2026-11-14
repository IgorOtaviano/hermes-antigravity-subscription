"""Regressions found reviewing the turn-home hardening change.

Each test pins one defect of that change:

* ``prepare_agy_home`` followed symlinks out of the isolated home, so a planted
  ``.gemini/config`` link turned the pre-spawn cleanup into a delete of the
  link's target.
* The lost-conversation retry ran after the failed stream had released the
  worker request lock, and told the retry stream it still held it.
* A continuation whose worker pipe was broken respawned a fresh conversation
  but kept expecting the old conversation id, so a healthy turn failed.
* A plan limit that ``AGY_ERROR`` reports only through its status reached the
  error classifier as plain text and was not classified.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from client import AntigravityClient
from process import AGY_PLANTED, harden_agy_settings, prepare_agy_home
from stream import AntigravityConversationLost, AntigravityStream


def _symlink_or_skip(test: unittest.TestCase, target: Path, link: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as exc:  # Windows without the privilege
        test.skipTest(f"symlinks unavailable: {exc}")


def _plant_everything(root: Path) -> None:
    """Create every AGY_PLANTED entry under *root* (a HOME-shaped directory)."""
    for rel in AGY_PLANTED:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith((".md", ".json")):
            target.write_text("user data", encoding="utf-8")
        else:
            target.mkdir(exist_ok=True)
            (target / "inside.txt").write_text("user data", encoding="utf-8")


class PlantedCleanupContainmentTests(unittest.TestCase):
    """The cleanup never deletes anything outside the isolated home."""

    def _home(self, root: Path) -> tuple[Path, Path]:
        home = root / "home"
        gemini = home / ".gemini" / "antigravity-cli"
        gemini.mkdir(parents=True)
        return home, gemini

    def test_a_planted_config_symlink_does_not_delete_its_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home, gemini = self._home(root)
            outside = root / "outside-config"
            (outside / "agents").mkdir(parents=True)
            (outside / "agents" / "mine.md").write_text("user data", encoding="utf-8")
            (outside / "config.json").write_text("user data", encoding="utf-8")
            (outside / "projects").mkdir()
            (outside / "projects" / "p.json").write_text("user data", encoding="utf-8")
            _symlink_or_skip(self, outside, home / ".gemini" / "config")

            prepare_agy_home(home, gemini)

            self.assertTrue((outside / "agents" / "mine.md").is_file())
            self.assertTrue((outside / "config.json").is_file())
            self.assertTrue((outside / "projects" / "p.json").is_file())
            # The link itself is planted config: agy would load its target.
            self.assertFalse(os.path.lexists(home / ".gemini" / "config"))
            self.assertTrue((gemini / "settings.json").is_file())

    def test_a_gemini_dir_linked_elsewhere_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            home.mkdir()
            real_gemini = root / "real-dot-gemini"
            real_gemini.mkdir()
            fake_real_home = root / "fake-real-home"
            (fake_real_home / ".gemini").parent.mkdir(parents=True)
            _symlink_or_skip(self, real_gemini, fake_real_home / ".gemini")
            _plant_everything(fake_real_home)
            _symlink_or_skip(self, real_gemini, home / ".gemini")
            gemini = home / ".gemini" / "antigravity-cli"
            gemini.mkdir(parents=True, exist_ok=True)

            prepare_agy_home(home, gemini)

            for rel in AGY_PLANTED:
                self.assertTrue(
                    os.path.lexists(fake_real_home / rel), f"{rel} was deleted through the link"
                )

    def test_the_real_home_is_never_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, gemini = self._home(Path(tmp))
            _plant_everything(home)
            with patch("process.Path.home", return_value=home):
                prepare_agy_home(home, gemini)
            for rel in AGY_PLANTED:
                self.assertTrue(os.path.lexists(home / rel), f"{rel} was deleted from the real home")

    def test_a_planted_leaf_symlink_is_unlinked_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home, gemini = self._home(root)
            outside = root / "outside-skills"
            outside.mkdir()
            (outside / "keep.md").write_text("user data", encoding="utf-8")
            (home / ".gemini" / "config").mkdir()
            _symlink_or_skip(self, outside, home / ".gemini" / "config" / "skills")

            prepare_agy_home(home, gemini)

            self.assertFalse(os.path.lexists(home / ".gemini" / "config" / "skills"))
            self.assertTrue((outside / "keep.md").is_file())

    def test_a_failed_settings_write_is_logged_as_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            _home, gemini = self._home(Path(tmp))
            with patch("process.os.replace", side_effect=OSError("disk full")):
                with self.assertLogs("process", level="WARNING") as logs:
                    harden_agy_settings(gemini)
            self.assertTrue(any("settings.json" in line for line in logs.output))


def _events(conversation_id: str, text: str) -> str:
    lines = [
        json.dumps({"event": "init", "conversation_id": conversation_id}),
        json.dumps({"event": "step_update", "step_update": {"text_delta": text}}),
        json.dumps(
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": text + "\n",
                    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                },
            }
        ),
    ]
    return "\n".join(lines) + "\n"


def _proc(stdout: object, *, alive: bool = False) -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = stdout
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None if alive else 0
    proc.wait.return_value = 0
    return proc


class _ProbedStdout:
    """A stdout whose every read records a probe (is the worker lock held?)."""

    def __init__(self, text: str, probe) -> None:
        self._inner = io.StringIO(text)
        self._probe = probe
        self.seen: list[bool] = []

    def readline(self, *args, **kwargs) -> str:
        self.seen.append(bool(self._probe()))
        return self._inner.readline(*args, **kwargs)

    def close(self) -> None:
        self._inner.close()


class ConversationRetryTests(unittest.TestCase):
    """The client-level paths around a resumed conversation."""

    def setUp(self):
        # Host facts must not leak in: the real token, the agy binary and, on
        # macOS, the tester's keychains (see test_antigravity_plugin.setUp).
        for patcher in (
            patch("client.is_authenticated", return_value=True),
            patch("client.resolve_real_token_path", return_value=None),
            patch("client.resolve_agy_command", return_value="agy"),
            patch("process._link_macos_keychains"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        # `agy models` runs through subprocess.Popen too. A fresh, empty catalog
        # keeps it from consuming one of the worker processes these tests queue.
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

    FIRST = [{"role": "user", "content": "first question"}]
    SECOND = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second question"},
    ]

    def test_lost_conversation_retry_holds_the_worker_lock(self):
        client = self.client
        proc_a = _proc(io.StringIO(_events("conv-1", "one")))
        proc_b = _proc(io.StringIO(_events("conv-OTHER", "contextless")))
        probed = _ProbedStdout(_events("conv-3", "three"), client._worker_lock.locked)
        proc_c = _proc(probed)

        with patch("subprocess.Popen", side_effect=[proc_a, proc_b, proc_c]) as popen:
            first = client.chat.completions.create(
                model="gemini-3.8-flash-high", messages=self.FIRST, stream=False
            )
            self.assertEqual(first.choices[0].message.content, "one")
            second = client.chat.completions.create(
                model="gemini-3.8-flash-high", messages=self.SECOND, stream=False
            )

        self.assertEqual(second.choices[0].message.content, "three")
        self.assertEqual(popen.call_count, 3)
        resumed_cmd = popen.call_args_list[1].args[0]
        retried_cmd = popen.call_args_list[2].args[0]
        self.assertIn("--conversation", resumed_cmd)
        self.assertNotIn("--conversation", retried_cmd)
        # The retry is a full send: the first exchange travels with it.
        sent = "".join(str(call.args[0]) for call in proc_c.stdin.write.call_args_list)
        self.assertIn("first question", sent)
        self.assertIn("second question", sent)
        # The retry turn owns the worker for its whole read, like any turn.
        self.assertTrue(probed.seen)
        self.assertTrue(all(probed.seen), f"lock not held during the retry: {probed.seen}")
        self.assertFalse(client._worker_lock.locked())

    def test_lost_conversation_retry_gives_up_when_the_worker_is_taken(self):
        client = self.client
        proc_a = _proc(io.StringIO(_events("conv-1", "one")))
        proc_b = _proc(io.StringIO(_events("conv-OTHER", "contextless")))
        real_release = AntigravityStream._release_worker_lock_once

        def release_then_let_another_turn_in(stream) -> None:
            real_release(stream)
            client._worker_lock.acquire(blocking=False)  # the other turn

        with patch("subprocess.Popen", side_effect=[proc_a, proc_b]) as popen, patch(
            "client._CONVERSATION_RETRY_LOCK_TIMEOUT", 0.05
        ):
            client.chat.completions.create(
                model="gemini-3.8-flash-high", messages=self.FIRST, stream=False
            )
            with patch.object(
                AntigravityStream,
                "_release_worker_lock_once",
                release_then_let_another_turn_in,
            ):
                with self.assertRaises(AntigravityConversationLost):
                    client.chat.completions.create(
                        model="gemini-3.8-flash-high", messages=self.SECOND, stream=False
                    )
        # No third spawn: the worker belongs to the other turn now.
        self.assertEqual(popen.call_count, 2)
        # And that turn's lock was not released from under it.
        self.assertTrue(client._worker_lock.locked())
        client._worker_lock.release()

    def test_broken_pipe_respawn_does_not_expect_the_old_conversation(self):
        client = self.client
        proc_a = _proc(io.StringIO(_events("conv-1", "one")), alive=True)
        proc_b = _proc(io.StringIO(_events("conv-2", "two")), alive=True)

        with patch("subprocess.Popen", side_effect=[proc_a, proc_b]) as popen:
            first = client.chat.completions.create(
                model="gemini-3.8-flash-high", messages=self.FIRST, stream=False
            )
            self.assertEqual(first.choices[0].message.content, "one")
            # The live worker's pipe is gone when the continuation is written.
            proc_a.stdin.write.side_effect = BrokenPipeError()
            stream = client.chat.completions.create(
                model="gemini-3.8-flash-high", messages=self.SECOND, stream=True
            )
            chunks = list(stream)

        text = "".join(
            choice.delta.content or ""
            for chunk in chunks
            for choice in getattr(chunk, "choices", [])
            if getattr(choice, "delta", None) is not None
        )
        self.assertEqual(text, "two")
        self.assertEqual(popen.call_count, 2)
        self.assertNotIn("--conversation", popen.call_args_list[1].args[0])


class PlanLimitStatusTests(unittest.TestCase):
    """A limit AGY_ERROR reports only by status still reaches the classifier."""

    def test_status_only_limit_is_classified_as_rate_limit(self):
        from providers import get_provider_profile

        with tempfile.TemporaryDirectory() as tmp:
            with patch("process._link_macos_keychains"):
                client = AntigravityClient(cwd=tmp)
            self.addCleanup(client.close)
            stream = AntigravityStream(
                proc=MagicMock(), client=client, model="gemini-3.8-flash", timeout=30.0
            )
            detail = stream._failure_detail(
                "AGY_ERROR: "
                + json.dumps(
                    {"status": "RESOURCE_EXHAUSTED", "code": 429, "message": "Individual limit reached"}
                ),
                "",
                "Process exited with return code 3",
            )
        self.assertIn("Individual limit reached", detail)

        profile = get_provider_profile("antigravity-subscription-directsdk")
        assert profile is not None and profile.classify_api_error is not None
        verdict = profile.classify_api_error(
            RuntimeError(f"Antigravity execution failed: {detail}")
        )
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["reason"], "rate_limit")

    def test_a_non_limit_failure_gains_no_limit_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("process._link_macos_keychains"):
                client = AntigravityClient(cwd=tmp)
            self.addCleanup(client.close)
            stream = AntigravityStream(
                proc=MagicMock(), client=client, model="gemini-3.8-flash", timeout=30.0
            )
            detail = stream._failure_detail(
                'AGY_ERROR: {"short_error": "model unavailable"}', "", "fallback"
            )
        self.assertEqual(detail, "model unavailable")


if __name__ == "__main__":
    unittest.main()
