"""New defects reproduced against the frozen module-one baseline."""
import io
import json
import os
import sys
import types
import unittest
from unittest.mock import patch

from tests.module1.support import ApiCase


class DefectCases(ApiCase):
    def test_a019_giant_numeric_input_is_validation_error(self):
        """A bounded numeric setting should reject even an unconvertible integer."""
        self.assert_rejected_without_record(maxScore=10 ** 400)

    def test_a020_normalized_duplicate_rubric_is_rejected(self):
        """Keywords that match the same normalized content should be one keyword."""
        self.assert_rejected_without_record(workContent='A', answerContent='AB',
            rubric=[{'keyword': 'A', 'weight': 1},
                    {'keyword': 'Ａ', 'weight': 1},
                    {'keyword': 'Z', 'weight': 1}])

    def test_a021_invalid_model_probability_cannot_be_hidden_by_rubric(self):
        """Validate the model's probability before mixing with a rubric."""
        fake = types.ModuleType('homework.models.transformer')
        fake.calculate_similarity = lambda work, answer: 2.0
        with patch.dict(sys.modules, {'homework.models.transformer': fake}):
            self.assert_rejected_without_record(expected=503, engine='transformer',
                workContent='A', answerContent='AB',
                rubric=[{'keyword': 'Z', 'weight': 1}])

    def test_a022_giant_ai_response_keeps_local_score(self):
        """An invalid cloud response should trigger the documented local fallback."""
        cloud_content = json.dumps({'score': 10 ** 400})
        response_body = json.dumps({'choices': [{'message': {'content': cloud_content}}]})
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only'}):
            with patch('urllib.request.urlopen', return_value=io.BytesIO(response_body.encode('utf-8'))):
                response = self.post(useDeepseek=True, aiWeight=70,
                    workContent='A', answerContent='AB')
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json['score'], 66.67)
        self.assertIsNone(response.json['aiPercent'])
        self.assertTrue(response.json['warnings'])


if __name__ == '__main__':
    unittest.main()
