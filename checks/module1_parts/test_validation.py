"""模块一输入校验测试（M1-UT-001～M1-UT-012）。"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from homework.validation import ValidationError, number, options, rubric, text


class ValidationTests(unittest.TestCase):
    def test_m1_ut_001_text_accepts_boundary_length(self):
        self.assertEqual(text("甲" * 5000, "作业内容"), "甲" * 5000)

    def test_m1_ut_002_text_rejects_over_boundary(self):
        with self.assertRaises(ValidationError):
            text("甲" * 5001, "作业内容")

    def test_m1_ut_003_text_trims_surrounding_spaces(self):
        self.assertEqual(text("  软件测试  ", "作业内容"), "软件测试")

    def test_m1_ut_004_required_text_rejects_blank(self):
        with self.assertRaises(ValidationError):
            text(" \n\t ", "作业内容")

    def test_m1_ut_005_number_accepts_minimum_and_maximum(self):
        self.assertEqual(number(1, "满分", 1, 1000), 1.0)
        self.assertEqual(number(1000, "满分", 1, 1000), 1000.0)

    def test_m1_ut_006_number_rejects_outside_range(self):
        for value in (0, 1001):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                number(value, "满分", 1, 1000)

    def test_m1_ut_007_number_rejects_non_numeric_and_non_finite_values(self):
        for value in (True, False, "100", None, float("nan"), float("inf")):
            with self.subTest(value=repr(value)), self.assertRaises(ValidationError):
                number(value, "满分", 1, 1000)

    def test_m1_ut_008_options_accept_valid_boundaries(self):
        value = options({"engine": "lexical", "maxScore": 1, "passPercent": 0,
                         "aiWeight": 100, "useDeepseek": False})
        self.assertEqual((value["maxScore"], value["passPercent"], value["aiWeight"]),
                         (1, 0, 100))
        for field in ("passPercent", "aiWeight"):
            for invalid in (-0.01, 100.01):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValidationError):
                    options({field: invalid})

    def test_m1_ut_009_options_reject_unknown_engine(self):
        with self.assertRaises(ValidationError):
            options({"engine": "unknown"})

    def test_m1_ut_010_rubric_accepts_twenty_items(self):
        items = [{"keyword": f"要点{i}", "weight": 1} for i in range(20)]
        self.assertEqual(len(rubric(items)), 20)

    def test_m1_ut_011_rubric_rejects_twenty_one_items(self):
        items = [{"keyword": f"要点{i}", "weight": 1} for i in range(21)]
        with self.assertRaises(ValidationError):
            rubric(items)

    def test_m1_ut_012_rubric_rejects_duplicate_keyword(self):
        with self.assertRaises(ValidationError):
            rubric([{"keyword": "缺陷", "weight": 1},
                    {"keyword": "缺陷", "weight": 2}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
