"""A-007..A-012: deterministic text scoring and observable interface results."""
import unittest

from homework.scoring import lexical_similarity, normalize
from tests.module1.support import ApiCase


class SimilarityCases(ApiCase):
    def test_a007_nfkc_case_and_punctuation_normalization(self):
        self.assertEqual(normalize('ＡＢＣ，abc!'), 'abcabc')
        response = self.post(workContent='ＡＢＣ！', answerContent='abc')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['score'], 100)

    def test_a008_dice_uses_character_multiplicity(self):
        self.assertAlmostEqual(lexical_similarity('aab', 'abb'), 2 / 3)
        response = self.post(workContent='aab', answerContent='abb')
        self.assertEqual(response.json['similarity'], 66.67)

    def test_a009_character_order_does_not_change_lexical_score(self):
        response = self.post(workContent='测试软件', answerContent='软件测试')
        self.assertEqual(response.json['score'], 100)
        self.assertIn('不能判断语义', response.json['explanation'])

    def test_a010_negation_is_documented_lexical_limit(self):
        response = self.post(workContent='我不喜欢', answerContent='我喜欢')
        self.assertEqual(response.json['similarity'], round(100 * 6 / 7, 2))
        self.assertIn('人工复核', response.json['explanation'])

    def test_a011_punctuation_only_answer_is_rejected(self):
        for field in ('workContent', 'answerContent'):
            with self.subTest(field=field):
                self.assert_rejected_without_record(**{field: '！？。'})

    def test_a012_result_contains_score_and_text_difference(self):
        response = self.post(workContent='A', answerContent='AB')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['score'], 66.67)
        self.assertEqual(response.json['basePercent'], 66.67)
        self.assertTrue(response.json['diff'])
        self.assertTrue(any(part['type'] != 'equal' for part in response.json['diff']))


if __name__ == '__main__':
    unittest.main()
