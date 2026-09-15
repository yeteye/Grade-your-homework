"""A-001..A-006: input equivalence classes and numeric boundaries."""
import json
import unittest

from tests.module1.support import ApiCase, PAYLOAD


class ValidationCases(ApiCase):
    def test_a001_missing_or_blank_answer_fields(self):
        for field in ('workContent', 'answerContent'):
            for value in (None, '', ' \t\n'):
                with self.subTest(field=field, value=repr(value)):
                    self.assert_rejected_without_record(**{field: value})

    def test_a002_text_at_5000_character_boundary(self):
        for field in ('workContent', 'answerContent'):
            with self.subTest(field=field):
                response = self.post(**{field: '字' * 5000})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(len(response.json[field]), 5000)

    def test_a003_text_over_5000_character_boundary(self):
        for field in ('workContent', 'answerContent'):
            with self.subTest(field=field):
                self.assert_rejected_without_record(**{field: '字' * 5001})

    def test_a004_max_score_boundaries(self):
        for value in (1, 1000):
            with self.subTest(valid=value):
                response = self.post(maxScore=value)
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(response.json['score'], value)
        for value in (0, -1, 1001):
            with self.subTest(invalid=value):
                self.assert_rejected_without_record(maxScore=value)

    def test_a005_percent_boundaries(self):
        for field in ('passPercent', 'aiWeight'):
            for value in (0, 100):
                with self.subTest(field=field, valid=value):
                    response = self.post(**{field: value})
                    self.assertEqual(response.status_code, 200, response.json)
                    self.assertEqual(response.json['options'][field], value)
            for value in (-0.01, 100.01):
                with self.subTest(field=field, invalid=value):
                    self.assert_rejected_without_record(**{field: value})

    def test_a006_non_numeric_or_non_finite_scores(self):
        for value in (True, False, '100', None, float('nan'), float('inf')):
            with self.subTest(value=repr(value)):
                self.assert_rejected_without_record(maxScore=value)


if __name__ == '__main__':
    unittest.main()
