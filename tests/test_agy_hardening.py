"""Turn-home hardening and CLI failure parsing, ported from agentes.chat.

The behaviors come from agentes.chat's `agy-home.ts` (the CLI's home:
AGY_PLANTED cleanup, settings) and `brain-antigravity.ts` (`AGY_ERROR`,
`finished_reason`, conversation-loss detection), ADR 0143. They are tested
here without a real `agy`: no subprocess, no account, no quota state.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from client import AntigravityClient
from process import (
    AGY_PLANTED,
    harden_agy_settings,
    parse_agy_error,
    prepare_agy_home,
)
from stream import AntigravityConversationLost, AntigravityStream


class ParseAgyErrorTests(unittest.TestCase):
    """`AGY_ERROR: {…}` is the CLI's own failure line (agy 1.2.x)."""

    def test_resource_exhausted_is_a_plan_limit_with_its_reset(self):
        stderr = (
            "some noise\n"
            "AGY_ERROR: "
            + json.dumps(
                {
                    "status": "RESOURCE_EXHAUSTED",
                    "code": 429,
                    "message": "Individual quota reached. Resets in 2h 30m.",
                }
            )
        )
        parsed = parse_agy_error(stderr)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertTrue(parsed["limit"])
        self.assertFalse(parsed["auth"])
        self.assertIn("Individual quota reached", parsed["message"])
        self.assertEqual(parsed["resets"], "2h 30m")

    def test_unauthenticated_is_an_auth_failure(self):
        stderr = "AGY_ERROR: " + json.dumps(
            {"canonical_status": "UNAUTHENTICATED", "http_status": 401, "message": "Please sign in"}
        )
        parsed = parse_agy_error(stderr)
        assert parsed is not None
        self.assertTrue(parsed["auth"])
        self.assertFalse(parsed["limit"])

    def test_a_non_json_line_keeps_the_cli_words(self):
        parsed = parse_agy_error("AGY_ERROR: weekly limit reached, resets at 2026-11-15T09:00Z")
        assert parsed is not None
        self.assertTrue(parsed["limit"])
        self.assertIn("weekly limit", parsed["message"])
        self.assertEqual(parsed["resets"], "2026-11-15T09:00Z")

    def test_no_line_is_none(self):
        self.assertIsNone(parse_agy_error(""))
        self.assertIsNone(parse_agy_error("plain stderr with no marker"))

    def test_the_158_short_error_shape_is_read(self):
        # agy 1.5.8's own format string: `AGY_ERROR: {"short_error":%q}`.
        parsed = parse_agy_error(
            'AGY_ERROR: {"short_error":"Individual quota reached. Resets in 3h."}'
        )
        assert parsed is not None
        self.assertIn("Individual quota reached", parsed["message"])
        self.assertTrue(parsed["limit"])
        self.assertEqual(parsed["resets"], "3h")

    def test_the_last_line_wins(self):
        stderr = "\n".join(
            [
                "AGY_ERROR: " + json.dumps({"message": "first"}),
                "AGY_ERROR: " + json.dumps({"message": "second"}),
            ]
        )
        parsed = parse_agy_error(stderr)
        assert parsed is not None
        self.assertEqual(parsed["message"], "second")


class HomeHardeningTests(unittest.TestCase):
    """`prepare_agy_home` + `harden_agy_settings` (agy-home.ts)."""

    def _home(self, root: Path) -> tuple[Path, Path]:
        home = root / "home"
        gemini = home / ".gemini" / "antigravity-cli"
        gemini.mkdir(parents=True)
        return home, gemini

    def test_settings_are_hardened_and_everything_else_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, gemini = self._home(Path(tmp))
            settings_path = gemini / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "auth": {"type": "oauth-personal"},
                        "gcp": {"project": "kept"},
                        "permissions": {"allow": ["read_file(/tmp/a.png)"]},
                    }
                ),
                encoding="utf-8",
            )
            harden_agy_settings(gemini)
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertFalse(settings["useG1Credits"])
            self.assertTrue(settings["enableTerminalSandbox"])
            self.assertFalse(settings["notifications"])
            self.assertFalse(settings["showTips"])
            self.assertFalse(settings["showFeedbackSurvey"])
            # Not ours: the sign-in, the seat's project and the media rules stay.
            self.assertEqual(settings["auth"], {"type": "oauth-personal"})
            self.assertEqual(settings["gcp"], {"project": "kept"})
            self.assertEqual(settings["permissions"]["allow"], ["read_file(/tmp/a.png)"])
            # Deliberately not forced: they change what works today.
            self.assertNotIn("toolPermission", settings)
            self.assertNotIn("allowNonWorkspaceAccess", settings)

    def test_credits_opt_in_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            _home, gemini = self._home(Path(tmp))
            with patch.dict("os.environ", {"ANTIGRAVITY_USE_CREDITS": "1"}):
                harden_agy_settings(gemini)
            settings = json.loads((gemini / "settings.json").read_text(encoding="utf-8"))
            self.assertTrue(settings["useG1Credits"])

    def test_settings_are_created_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            _home, gemini = self._home(Path(tmp))
            harden_agy_settings(gemini)
            self.assertTrue((gemini / "settings.json").is_file())

    def test_planted_config_is_removed_and_user_files_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, gemini = self._home(Path(tmp))
            for rel in AGY_PLANTED:
                target = home / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if rel.endswith(".md") or rel.endswith(".json"):
                    target.write_text("planted", encoding="utf-8")
                else:
                    target.mkdir(exist_ok=True)
            keep = home / ".gemini" / "config" / "unrelated.txt"
            keep.parent.mkdir(parents=True, exist_ok=True)
            keep.write_text("mine", encoding="utf-8")

            prepare_agy_home(home, gemini)

            for rel in AGY_PLANTED:
                self.assertFalse((home / rel).exists(), f"{rel} survived")
            self.assertTrue(keep.is_file())
            self.assertTrue((gemini / "settings.json").is_file())


class SandboxFlagTests(unittest.TestCase):
    """`--sandbox` on every spawn, the flag agentes.chat always passes."""

    def test_default_args_carry_sandbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AntigravityClient(cwd=tmp)
            self.addCleanup(client.close)
            self.assertIn("--sandbox", client._args)
            self.assertNotIn("--dangerously-skip-permissions", client._args)

    def test_explicit_args_still_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AntigravityClient(cwd=tmp, args=["--output-format", "stream-json"])
            self.addCleanup(client.close)
            self.assertEqual(client._args, ["--output-format", "stream-json"])


class StreamFailureTests(unittest.TestCase):
    """Failure wording and conversation-loss detection (brain-antigravity.ts)."""

    def _stream(self, client: AntigravityClient, proc: MagicMock, **kwargs) -> AntigravityStream:
        return AntigravityStream(
            proc=proc,
            client=client,
            model="gemini-3.8-flash",
            timeout=30.0,
            **kwargs,
        )

    def test_failure_detail_reads_the_agy_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AntigravityClient(cwd=tmp)
            self.addCleanup(client.close)
            stream = self._stream(client, MagicMock())
            detail = stream._failure_detail(
                "AGY_ERROR: "
                + json.dumps(
                    {
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "Individual quota reached. Resets in 2h 30m.",
                    }
                ),
                "",
                "Process exited with return code 3",
            )
            self.assertIn("Individual quota reached", detail)
            self.assertIn("resets 2h 30m", detail)
            self.assertNotIn("AGY_ERROR", detail)
            self.assertNotIn("return code", detail)

    def test_failure_detail_falls_back_without_the_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AntigravityClient(cwd=tmp)
            self.addCleanup(client.close)
            stream = self._stream(client, MagicMock())
            self.assertEqual(
                stream._failure_detail("", "", "Process exited with return code 3"),
                "Process exited with return code 3",
            )

    def test_print_timeout_reads_as_a_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AntigravityClient(cwd=tmp)
            self.addCleanup(client.close)
            stream = self._stream(client, MagicMock())
            message = stream._empty_result_message(
                saw_result=True,
                status="INCOMPLETE",
                process_exit=0,
                has_evidence=False,
                finished_reason="print_timeout",
            )
            self.assertIn("ran out of time", message)

    def test_a_resume_answered_by_another_conversation_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AntigravityClient(cwd=tmp)
            self.addCleanup(client.close)
            proc = MagicMock()
            proc.poll.return_value = None
            proc.stdin = MagicMock()
            events = [
                json.dumps({"event": "init", "conversation_id": "a-different-conv"}),
                json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "hi"}}),
            ]
            proc.stdout.readline.side_effect = [f"{line}\n" for line in events] + [""]
            with patch.object(client, "_terminate_process"):
                stream = self._stream(
                    client, proc, expected_conversation_id="the-conv-we-asked-for"
                )
                with self.assertRaises(AntigravityConversationLost):
                    list(stream)

    def test_a_resume_answered_by_the_same_conversation_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AntigravityClient(cwd=tmp)
            self.addCleanup(client.close)
            proc = MagicMock()
            proc.poll.return_value = 0
            proc.stdin = MagicMock()
            events = [
                json.dumps({"event": "init", "conversation_id": "conv-1"}),
                json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "ok"}}),
            ]
            proc.stdout.readline.side_effect = [f"{line}\n" for line in events] + [""]
            with patch.object(client, "_terminate_process"):
                stream = self._stream(client, proc, expected_conversation_id="conv-1")
                chunks = list(stream)
            self.assertTrue(chunks)


class PlanLimitClassificationTests(unittest.TestCase):
    """A quota failure is a rate limit, not an unclassifiable error."""

    def test_quota_is_classified_as_rate_limit(self):
        from providers import get_provider_profile

        profile = get_provider_profile("antigravity-subscription-directsdk")
        assert profile is not None and profile.classify_api_error is not None
        hook = profile.classify_api_error
        verdict = hook(
            RuntimeError(
                "Antigravity execution failed: Individual quota reached. Resets in 2h 30m."
            )
        )
        self.assertEqual(verdict["reason"], "rate_limit")
        self.assertTrue(verdict["retryable"])
        self.assertTrue(verdict["should_fallback"])

    def test_billing_still_wins_over_quota(self):
        from providers import get_provider_profile

        profile = get_provider_profile("antigravity-subscription-directsdk")
        assert profile is not None and profile.classify_api_error is not None
        hook = profile.classify_api_error
        verdict = hook(
            RuntimeError("Antigravity execution failed: Your AI credits balance is too low")
        )
        self.assertEqual(verdict["reason"], "billing")


if __name__ == "__main__":
    unittest.main()
