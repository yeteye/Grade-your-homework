import tempfile
import unittest
from unittest.mock import patch

from server import create_app
from homework.storage import DEFAULTS


PAYLOAD = dict(studentName='测试学生A', title='模块一验证',
               workContent='软件测试', answerContent='软件测试')


class ApiCase(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='module1-case-')
        self.addCleanup(self.folder.cleanup)
        self.app = create_app({'TESTING': True, 'DATA_DIR': self.folder.name})
        self.client = self.app.test_client()
        # A real cloud call is never an acceptable side effect of course tests.
        self.network = patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network call'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def post(self, **changes):
        return self.client.post('/compare_texts', json={**PAYLOAD, **changes})

    def assert_rejected_without_record(self, expected=400, **changes):
        before = self.client.get('/api/stats').json['total']
        response = self.post(**changes)
        self.assertEqual(response.status_code, expected, response.json)
        self.assertIn('message', response.json)
        self.assertEqual(self.client.get('/api/stats').json['total'], before)
        return response
