"""Prompt budget: a large turn keeps its tail (issue #29).

agy truncates an oversized single prompt from its tail, which silently drops the
latest user request and the tool-loop instructions at the end of the prompt. The
budget here is the one agentes-chat applies to its cursor prompt: a hard
character limit with MIDDLE elision, so the head (system preamble, tool schemas)
and the tail (the request the model must answer) always survive.
"""

from __future__ import annotations

import unittest

import prompt as agy_prompt

TAIL_REQUEST = "ULTIMO PEDIDO: rode o comando X e responda com o resultado"


def _messages(turns: int, chars_per_turn: int = 6000) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "SYS: you are Hermes."}]
    for i in range(turns):
        messages.append(
            {"role": "user", "content": f"U{i}: " + ("pedido " * (chars_per_turn // 7 + 1))}
        )
        messages.append(
            {"role": "assistant", "content": f"A{i}: " + ("resposta " * (chars_per_turn // 9 + 1))}
        )
    messages.append({"role": "user", "content": TAIL_REQUEST})
    return messages


def _tools(count: int = 40) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "d" * 400,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for i in range(count)
    ]


class PromptBudgetTests(unittest.TestCase):
    def test_small_prompt_is_untouched(self):
        assembled = agy_prompt._format_messages_as_prompt(_messages(1, 200), tools=_tools(3))
        self.assertNotIn("elided", assembled)
        self.assertIn(TAIL_REQUEST, assembled)

    def test_large_prompt_fits_the_budget_and_keeps_the_tail(self):
        assembled = agy_prompt._format_messages_as_prompt(_messages(60), tools=_tools())
        self.assertLessEqual(len(assembled), agy_prompt._PROMPT_MAX_CHARS)
        self.assertIn(TAIL_REQUEST, assembled)
        self.assertIn("transcript blocks elided", assembled)

    def test_head_and_tail_of_the_transcript_survive(self):
        assembled = agy_prompt._format_messages_as_prompt(_messages(60), tools=_tools())
        self.assertIn("U0:", assembled)  # head of the transcript
        self.assertIn("A59:", assembled)  # tail of the transcript

    def test_budget_can_be_disabled(self):
        assembled = agy_prompt._fit_prompt_sections(["x" * 500], limit=0)
        self.assertEqual(assembled, ["x" * 500])

    def test_budget_under_limit_returns_the_same_list(self):
        sections = ["a" * 100, "b" * 100]
        self.assertEqual(agy_prompt._fit_prompt_sections(sections, limit=1000), sections)

    def test_pinned_instruction_sections_are_never_elided(self):
        pinned = "### ACTIVE OBJECTIVE:\n" + "p" * 5000
        sections = [pinned, "filler " * 30000]
        fitted = agy_prompt._fit_prompt_sections(sections, limit=4000)
        self.assertIn(pinned, fitted)

    def test_elide_blocks_keeps_both_ends_and_reports_the_cut(self):
        blocks = [f"block-{i}-" + "x" * 100 for i in range(20)]
        kept, head_count, dropped, dropped_chars = agy_prompt._elide_blocks(blocks, room=1200)
        self.assertGreater(head_count, 0)
        self.assertGreater(dropped, 0)
        self.assertGreater(dropped_chars, 0)
        self.assertEqual(kept[0], blocks[0])
        self.assertEqual(kept[-1], blocks[-1])
        self.assertLessEqual(len(kept), len(blocks))


if __name__ == "__main__":
    unittest.main()
