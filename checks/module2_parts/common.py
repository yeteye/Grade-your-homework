"""Shared fixtures for the Module 2 AI-assisted test scripts."""
import tempfile
import unittest

from server import create_app


BASE_PAYLOAD = {
    "studentName": "模块二同学",
    "title": "模块二接口测试",
    "workContent": "软件测试用于发现缺陷",
    "answerContent": "软件测试用于发现缺陷",
    "engine": "lexical",
    "maxScore": 100,
    "passPercent": 60,
    "aiWeight": 70,
    "useDeepseek": False,
    "rubric": [],
}


class ApiCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app({"TESTING": True, "DATA_DIR": self.temp.name})
        self.client = self.app.test_client()
        self.store = self.app.extensions["store"]

    def tearDown(self):
        self.temp.cleanup()

    def create_record(self, **changes):
        payload = {**BASE_PAYLOAD, **changes}
        response = self.client.post("/compare_texts", json=payload)
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()
