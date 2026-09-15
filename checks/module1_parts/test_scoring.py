"""模块一评分、AI 回退与 Transformer 边界测试（M1-UT-013～022）。"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from homework.ai_grader import GradingUnavailable, parse_score
from homework.scoring import compare, lexical_similarity, normalize
from homework.validation import ValidationError

DEFAULTS = {"engine": "lexical", "maxScore": 100, "passPercent": 60,
            "aiWeight": 70, "useDeepseek": False}


class ScoringTests(unittest.TestCase):
    def payload(self, **changes):
        value = {"studentName": "测试同学", "title": "模块一",
                 "workContent": "软件测试发现缺陷", "answerContent": "软件测试发现缺陷"}
        value.update(changes)
        return value

    def test_m1_ut_013_normalize_unifies_width_case_and_punctuation(self):
        self.assertEqual(normalize("ＡbＣ！"), "abc")

    def test_m1_ut_014_lexical_identical_text_scores_one(self):
        self.assertEqual(lexical_similarity("软件测试", "软件测试"), 1)

    def test_m1_ut_015_lexical_empty_normalized_text_scores_zero(self):
        self.assertEqual(lexical_similarity("!!!", "软件测试"), 0)

    def test_m1_ut_016_weighted_rubric_changes_score(self):
        result = compare(self.payload(workContent="软件测试", rubric=[
            {"keyword": "软件", "weight": 1}, {"keyword": "缺陷", "weight": 1}]), DEFAULTS)
        expected = 0.4 * lexical_similarity("软件测试", "软件测试发现缺陷") + 0.6 * 0.5
        self.assertAlmostEqual(result["similarity"], round(expected * 100, 2))

    def test_m1_ut_017_displayed_score_controls_pass_boundary(self):
        result = compare(self.payload(maxScore=10, passPercent=100), DEFAULTS)
        self.assertEqual(result["score"], 10)
        self.assertTrue(result["passed"])

    def test_m1_ut_018_ai_failure_preserves_base_score(self):
        with patch("homework.scoring.get_points", side_effect=GradingUnavailable("模拟超时")):
            result = compare(self.payload(useDeepseek=True), DEFAULTS)
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["warnings"], ["模拟超时"])

    def test_m1_ut_019_ai_valid_score_is_weighted(self):
        with patch("homework.scoring.get_points", return_value=0.5):
            result = compare(self.payload(useDeepseek=True, aiWeight=40), DEFAULTS)
        self.assertEqual(result["score"], 80)

    def test_m1_ut_020_ai_response_rejects_invalid_values(self):
        for value in ("-0.1", "1.1", "NaN", "true", '{"score":"0.5"}'):
            with self.subTest(value=value), self.assertRaises(GradingUnavailable):
                parse_score(value)

    def test_m1_ut_021_transformer_accepts_combined_length_509(self):
        fake = types.ModuleType("homework.models.transformer")
        fake.calculate_similarity = lambda _work, _answer: 0.8
        with patch.dict(sys.modules, {"homework.models.transformer": fake}):
            result = compare(self.payload(workContent="甲" * 254, answerContent="乙" * 255,
                                          engine="transformer"), DEFAULTS)
        self.assertEqual(result["score"], 80)

    def test_m1_ut_022_transformer_rejects_combined_length_510(self):
        with self.assertRaises(ValidationError):
            compare(self.payload(workContent="甲" * 255, answerContent="乙" * 255,
                                 engine="transformer"), DEFAULTS)

    def test_m1_ut_040_dice_counts_repeated_characters(self):
        self.assertAlmostEqual(lexical_similarity("aab", "abb"), 2 / 3)

    def test_m1_ut_041_character_order_does_not_change_lexical_score(self):
        result = compare(self.payload(workContent="测试软件", answerContent="软件测试"), DEFAULTS)
        self.assertEqual(result["score"], 100)
        self.assertIn("不能判断语义", result["explanation"])

    def test_m1_ut_042_negation_requires_manual_review(self):
        result = compare(self.payload(workContent="我不喜欢", answerContent="我喜欢"), DEFAULTS)
        self.assertEqual(result["similarity"], round(100 * 6 / 7, 2))
        self.assertIn("人工复核", result["explanation"])

    def test_m1_ut_043_punctuation_only_answer_is_rejected(self):
        for field in ("workContent", "answerContent"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                compare(self.payload(**{field: "！？。"}), DEFAULTS)

    def test_m1_ut_044_partial_match_returns_score_and_diff(self):
        result = compare(self.payload(workContent="A", answerContent="AB"), DEFAULTS)
        self.assertEqual(result["score"], 66.67)
        self.assertEqual(result["basePercent"], 66.67)
        self.assertTrue(any(part["type"] != "equal" for part in result["diff"]))

    def test_m1_ut_045_rubric_keyword_length_boundaries(self):
        self.assertEqual(len(compare(self.payload(rubric=[{"keyword": "词" * 80, "weight": 1}]), DEFAULTS)["rubric"]), 1)
        with self.assertRaises(ValidationError):
            compare(self.payload(rubric=[{"keyword": "词" * 81, "weight": 1}]), DEFAULTS)

    def test_m1_ut_046_rubric_weight_boundaries(self):
        for value in (0.1, 100):
            with self.subTest(valid=value):
                compare(self.payload(rubric=[{"keyword": "软件", "weight": value}]), DEFAULTS)
        for value in (0, 0.09, 100.01, True):
            with self.subTest(invalid=value), self.assertRaises(ValidationError):
                compare(self.payload(rubric=[{"keyword": "软件", "weight": value}]), DEFAULTS)

    def test_m1_ut_047_uneven_rubric_weights_drive_score(self):
        result = compare(self.payload(workContent="甲", answerContent="甲乙", rubric=[
            {"keyword": "甲", "weight": 3}, {"keyword": "乙", "weight": 1}]), DEFAULTS)
        self.assertEqual(result["basePercent"], 71.67)
        self.assertEqual(result["score"], 71.67)

    def test_m1_ut_048_displayed_score_controls_fractional_threshold(self):
        accepted = compare(self.payload(workContent="A", answerContent="AB", passPercent=66.67), DEFAULTS)
        rejected = compare(self.payload(workContent="A", answerContent="AB", passPercent=66.68), DEFAULTS)
        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
