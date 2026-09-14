"""Developer regression checks for this change, not course submission materials."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from server import create_app
from homework.ai_grader import parse_score, GradingUnavailable
from homework.scoring import lexical_similarity


def png(color):
    stream = io.BytesIO()
    Image.new('RGB', (16, 16), color).save(stream, 'PNG')
    stream.seek(0)
    return stream


class RegressionChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.app = create_app({'TESTING': True, 'DATA_DIR': self.folder.name})
        self.client = self.app.test_client()
        self.payload = dict(studentName='示例同学', title='练习', workContent='软件测试发现缺陷', answerContent='软件测试发现缺陷')

    def tearDown(self):
        self.folder.cleanup()

    def test_persist_and_export_actual_result(self):
        r = self.client.post('/compare_texts', json=self.payload)
        self.assertEqual(r.status_code, 200)
        result = r.get_json()
        self.assertEqual(result['score'], 100)
        record = self.client.get('/api/records/' + result['id']).get_json()
        self.assertEqual(record['workContent'], self.payload['workContent'])
        csv = self.client.get('/api/export')
        self.assertIn('示例同学', csv.data.decode('utf-8-sig'))
        self.assertIn('attachment', csv.headers['Content-Disposition'])
        self.assertEqual(self.client.get('/api/stats').get_json()['total'], 1)
        self.assertEqual(self.client.delete('/api/records/' + result['id']).status_code, 200)
        self.assertEqual(self.client.get('/api/stats').get_json()['total'], 0)

    def test_ai_failure_keeps_base_score(self):
        with patch('homework.scoring.get_points', side_effect=GradingUnavailable('模拟超时')):
            r = self.client.post('/compare_texts', json={**self.payload, 'useDeepseek': True}).get_json()
        self.assertEqual(r['score'], 100)
        self.assertIsNone(r['aiPercent'])
        self.assertEqual(r['warnings'], ['模拟超时'])

    def test_ai_score_validation(self):
        for value in ('-0.1', '1.1', 'NaN', 'Infinity', 'true', '{"score":"0.5"}', '评分为0.8'):
            with self.subTest(value=value), self.assertRaises(GradingUnavailable):
                parse_score(value)
        self.assertEqual(parse_score('{"score":0.75}'), 0.75)

    def test_parameter_validation(self):
        for body in (None, [], 'hello', {**self.payload, 'workContent': ['x']},
                     {**self.payload, 'workContent': '  '}, {**self.payload, 'maxScore': True},
                     {**self.payload, 'maxScore': 0}, {**self.payload, 'useDeepseek': 'false'},
                     {**self.payload, 'workContent': '字' * 5001}):
            with self.subTest(body_type=type(body).__name__):
                r = self.client.post('/compare_texts', data=json.dumps(body), content_type='application/json')
                self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.get('/api/stats').get_json()['total'], 0)

    def test_batch_all_or_nothing(self):
        data = {**self.payload, 'items': [{'studentName':'甲','workContent':'正确'}, {'studentName':'乙','workContent':''}]}
        self.assertEqual(self.client.post('/api/batch', json=data).status_code, 400)
        self.assertEqual(self.client.get('/api/stats').get_json()['total'], 0)
        data['items'][1]['workContent'] = '有效文本'
        self.assertEqual(self.client.post('/api/batch', json=data).status_code, 201)
        self.assertEqual(self.client.get('/api/stats').get_json()['total'], 2)

    def test_same_name_images_are_isolated_and_cleaned(self):
        observed = []
        def recognize(path, engine, language):
            with Image.open(path) as image:
                observed.append(image.getpixel((0, 0)))
            return '已识别文字'
        with patch('homework.ocr.status', return_value={'RapidOCR': True}), patch('homework.ocr.recognize', side_effect=recognize):
            r = self.client.post('/ocr', data={'file1':(png('red'),'same.png'), 'file2':(png('blue'),'same.png')})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(observed, [(255,0,0), (0,0,255)])
        self.assertEqual(list((Path(self.folder.name) / 'temp').iterdir()), [])

    def test_path_name_is_never_used_for_writing(self):
        with patch('homework.ocr.status', return_value={'RapidOCR': True}), patch('homework.ocr.recognize', return_value='文字'):
            r = self.client.post('/ocr', data={'file1':(png('white'),'../../escaped.png'), 'file2':(png('white'),'b.png')})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(list(Path(self.folder.name).glob('*.png')), [])

    def test_invalid_image_fails_before_engine_load(self):
        with patch('homework.ocr.recognize') as engine:
            r = self.client.post('/ocr', data={'file1':(io.BytesIO(b'not an image'),'a.png'),'file2':(png('white'),'b.png')})
            self.assertEqual(r.status_code, 400)
            engine.assert_not_called()
        self.assertEqual(list((Path(self.folder.name)/'temp').iterdir()), [])

    def test_ocr_failure_is_json_and_cleans_files(self):
        with patch('homework.ocr.status', return_value={'RapidOCR': True}), patch('homework.ocr.recognize', side_effect=RuntimeError()):
            r = self.client.post('/ocr', data={'file1':(png('red'),'a.png'),'file2':(png('blue'),'b.png')})
        self.assertEqual(r.status_code, 503)
        self.assertIn('message', r.get_json())
        self.assertEqual(list((Path(self.folder.name)/'temp').iterdir()), [])

    def test_template_and_settings_persist_across_app_restart(self):
        template = {'title':'模板', 'answerContent':'参考文本','rubric':[{'keyword':'参考','weight':2}]}
        r = self.client.post('/api/templates', json=template)
        self.assertEqual(r.status_code, 201)
        identifier = r.get_json()['id']
        template['title'] = '更新模板'
        self.assertEqual(self.client.put('/api/templates/'+identifier, json=template).status_code, 200)
        self.client.put('/api/settings', json={'maxScore':20})
        other = create_app({'TESTING':True,'DATA_DIR':self.folder.name}).test_client()
        self.assertEqual(other.get('/api/settings').get_json()['maxScore'], 20)
        self.assertEqual(other.get('/api/templates').get_json()['items'][0]['title'], '更新模板')
        self.assertEqual(other.delete('/api/templates/'+identifier).status_code, 200)

    def test_long_model_input_rejected_before_loading(self):
        r = self.client.post('/compare_texts', json={**self.payload,'engine':'transformer','workContent':'字'*510})
        self.assertEqual(r.status_code, 400)

    def test_csv_formula_and_html_are_escaped(self):
        r = self.client.post('/compare_texts', json={**self.payload,'studentName':'=1+1','workContent':'<script>alert(1)</script>'}).get_json()
        self.assertIn("'=1+1", self.client.get('/api/export').data.decode('utf-8-sig'))
        html = self.client.get('/records/'+r['id']+'/print').data.decode()
        self.assertNotIn('<script>alert', html)
        self.assertIn('&lt;script&gt;', html)

    def test_filter_and_invalid_pagination(self):
        self.client.post('/compare_texts', json=self.payload)
        self.assertEqual(self.client.get('/api/records?q=不存在').get_json()['total'], 0)
        self.assertEqual(self.client.get('/api/records?status=passed').get_json()['total'], 1)
        for query in ('page=0','page=NaN','pageSize=101','status=unknown'):
            self.assertEqual(self.client.get('/api/records?'+query).status_code, 400)

    def test_same_origin_guard(self):
        self.assertEqual(self.client.post('/compare_texts', json=self.payload, headers={'Origin':'https://example.com'}).status_code, 403)

    def test_normalization(self):
        self.assertEqual(lexical_similarity('ＡＢＣ！', 'abc'), 1)
        self.assertEqual(lexical_similarity('  ', 'abc'), 0)


if __name__ == '__main__':
    unittest.main()
