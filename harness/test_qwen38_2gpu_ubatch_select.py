#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

path = pathlib.Path(__file__).with_name("qwen38_2gpu_ubatch_select.py")
spec = importlib.util.spec_from_file_location("ubsel", path)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def rows(label, case, pp, dec, n=6, code=200):
    return [{"label": label, "case": case, "prompt_tps": pp, "decode_tps": dec, "http_code": code} for _ in range(n)]


class UbatchSelectorTests(unittest.TestCase):
    def baseline(self):
        out = []
        for case, pp in (("short", 125), ("p4k", 575), ("p32k", 500)):
            out += rows("ub512", case, pp, 35)
        return out

    def test_best_weighted_long_prompt_candidate_wins(self):
        data = self.baseline()
        for case, pp in (("short", 120), ("p4k", 650), ("p32k", 570)):
            data += rows("ub320", case, pp, 35)
        for case, pp in (("short", 130), ("p4k", 600), ("p32k", 530)):
            data += rows("ub384", case, pp, 35)
        winner, _ = mod.select(mod.summarize(data))
        self.assertEqual(winner, "ub320")

    def test_decode_regression_rejects_candidate(self):
        data = self.baseline()
        for case, pp in (("short", 130), ("p4k", 700), ("p32k", 650)):
            data += rows("ub320", case, pp, 30)
        winner, checked = mod.select(mod.summarize(data))
        self.assertIsNone(winner)
        self.assertFalse(checked["ub320"]["eligible"])

    def test_incomplete_case_rejects_candidate(self):
        data = self.baseline()
        data += rows("ub320", "short", 130, 35)
        data += rows("ub320", "p4k", 650, 35)
        data += rows("ub320", "p32k", 570, 35, n=5)
        winner, checked = mod.select(mod.summarize(data))
        self.assertIsNone(winner)
        self.assertFalse(checked["ub320"]["eligible"])


if __name__ == "__main__":
    unittest.main()
