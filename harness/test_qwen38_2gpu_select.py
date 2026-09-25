#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

PATH = pathlib.Path(__file__).with_name("qwen38_2gpu_select.py")
spec = importlib.util.spec_from_file_location("selector", PATH)
assert spec is not None and spec.loader is not None
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def row(label, pp, dec, gpu0, gpu1, code=200):
    return {
        "label": label,
        "http_code": code,
        "prompt_tps": pp,
        "decode_tps": dec,
        "gpu_after": [f"0, {gpu0}, P0", f"1, {gpu1}, P0"],
    }


class SelectorTests(unittest.TestCase):
    def test_balanced_candidate_wins_within_performance_gate(self):
        rows = []
        for _ in range(3):
            rows += [row("split100", 500, 35, 15500, 18800), row("split115", 505, 35, 17300, 17400)]
        summary = selector.summarize(rows, expected=3)
        winner, _ = selector.select(summary)
        self.assertEqual(winner, "split115")

    def test_incomplete_candidate_is_rejected(self):
        rows = [row("split100", 500, 35, 15500, 18800) for _ in range(3)]
        rows += [row("split115", 520, 36, 17300, 17400) for _ in range(2)]
        summary = selector.summarize(rows, expected=3)
        winner, checked = selector.select(summary)
        self.assertEqual(winner, "split100")
        self.assertFalse(checked["split115"]["eligible"])

    def test_fast_but_decode_regressed_candidate_is_rejected(self):
        rows = []
        for _ in range(3):
            rows += [row("split100", 500, 35, 15500, 18800), row("split120", 600, 33, 17200, 17300)]
        summary = selector.summarize(rows, expected=3)
        winner, checked = selector.select(summary)
        self.assertEqual(winner, "split100")
        self.assertFalse(checked["split120"]["eligible"])

    def test_http_failure_makes_arm_incomplete(self):
        rows = [row("split100", 500, 35, 15500, 18800) for _ in range(3)]
        rows += [row("split115", 505, 35, 17300, 17400) for _ in range(2)]
        rows += [row("split115", None, None, 17300, 17400, code=500)]
        summary = selector.summarize(rows, expected=3)
        winner, checked = selector.select(summary)
        self.assertEqual(winner, "split100")
        self.assertFalse(checked["split115"]["eligible"])


if __name__ == "__main__":
    unittest.main()
