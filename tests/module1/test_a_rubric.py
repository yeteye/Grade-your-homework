"""A-013..A-018: rubric size, weights, thresholds and AI composition."""
import unittest
from unittest.mock import patch

from homework.ai_grader import GradingUnavailable
from tests.module1.support import ApiCase


class RubricCases(ApiCase):
    def test_a013_rubric_count_boundaries(self):
        for size in (0, 1, 20):
            points = [{'keyword': f'点{i}', 'weight': 1} for i in range(size)]
            with self.subTest(valid=size):
                response = self.post(rubric=points)
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(len(response.json['rubric']), size)
        points = [{'keyword': f'点{i}', 'weight': 1} for i in range(21)]
        self.assert_rejected_without_record(rubric=points)

    def test_a014_keyword_length_boundaries(self):
        response = self.post(rubric=[{'keyword': '词' * 80, 'weight': 1}])
        self.assertEqual(response.status_code, 200, response.json)
        self.assert_rejected_without_record(rubric=[{'keyword': '词' * 81, 'weight': 1}])

    def test_a015_keyword_weight_boundaries(self):
        for value in (0.1, 100):
            with self.subTest(valid=value):
                self.assertEqual(self.post(rubric=[{'keyword': '软件', 'weight': value}]).status_code, 200)
        for value in (0, 0.09, 100.01, True):
            with self.subTest(invalid=value):
                self.assert_rejected_without_record(rubric=[{'keyword': '软件', 'weight': value}])

    def test_a016_weighted_coverage_drives_base_score(self):
        response = self.post(workContent='甲', answerContent='甲乙',
            rubric=[{'keyword': '甲', 'weight': 3}, {'keyword': '乙', 'weight': 1}])
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual([r['matched'] for r in response.json['rubric']], [True, False])
        self.assertEqual(response.json['basePercent'], 71.67)
        self.assertEqual(response.json['score'], 71.67)

    def test_a017_pass_threshold_uses_displayed_score(self):
        accepted = self.post(workContent='A', answerContent='AB', passPercent=66.67)
        rejected = self.post(workContent='A', answerContent='AB', passPercent=66.68)
        self.assertEqual(accepted.json['score'], 66.67)
        self.assertTrue(accepted.json['passed'])
        self.assertFalse(rejected.json['passed'])

    def test_a018_ai_weight_and_failure_fallback(self):
        with patch('homework.scoring.get_points', return_value=0.5):
            weighted = self.post(useDeepseek=True, aiWeight=20)
        self.assertEqual(weighted.status_code, 200, weighted.json)
        self.assertEqual(weighted.json['score'], 90)
        self.assertEqual(weighted.json['aiPercent'], 50)
        with patch('homework.scoring.get_points', side_effect=GradingUnavailable('模拟超时')):
            fallback = self.post(useDeepseek=True, aiWeight=20)
        self.assertEqual(fallback.status_code, 200, fallback.json)
        self.assertEqual(fallback.json['score'], 100)
        self.assertEqual(fallback.json['warnings'], ['模拟超时'])


if __name__ == '__main__':
    unittest.main()
