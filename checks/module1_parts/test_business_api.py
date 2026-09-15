"""模块一批改、批量、设置、模板、记录与缺陷回归接口测试。"""
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server import create_app


class BusinessApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.client = create_app({"TESTING": True, "DATA_DIR": self.folder.name}).test_client()
        self.payload = {"studentName": "示例同学", "title": "模块一",
                        "workContent": "软件测试发现缺陷", "answerContent": "软件测试发现缺陷"}

    def tearDown(self):
        self.folder.cleanup()

    def test_m1_it_036_compare_persists_record(self):
        created = self.client.post("/compare_texts", json=self.payload)
        self.assertEqual(created.status_code, 200)
        record = self.client.get("/api/records/" + created.get_json()["id"])
        self.assertEqual(record.get_json()["workContent"], self.payload["workContent"])

    def test_m1_it_037_invalid_compare_does_not_persist(self):
        response = self.client.post("/compare_texts", json={**self.payload, "workContent": " "})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/stats").get_json()["total"], 0)

    def test_m1_it_038_batch_accepts_one_and_twenty_items(self):
        for count in (1, 20):
            with self.subTest(count=count):
                data = {**self.payload, "useDeepseek": False,
                        "items": [{"studentName": f"学生{i}", "workContent": "正确"}
                                  for i in range(count)]}
                response = self.client.post("/api/batch", json=data)
                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.get_json()["count"], count)

    def test_m1_it_039_batch_rejects_zero_and_twenty_one_items(self):
        for count in (0, 21):
            with self.subTest(count=count):
                response = self.client.post("/api/batch", json={**self.payload, "items": [{}] * count})
                self.assertEqual(response.status_code, 400)

    def test_m1_it_040_batch_failure_is_atomic(self):
        data = {**self.payload, "items": [
            {"studentName": "甲", "workContent": "有效"},
            {"studentName": "乙", "workContent": ""}]}
        self.assertEqual(self.client.post("/api/batch", json=data).status_code, 400)
        self.assertEqual(self.client.get("/api/stats").get_json()["total"], 0)

    def test_m1_it_041_settings_persist_after_restart(self):
        self.assertEqual(self.client.put("/api/settings", json={"maxScore": 20}).status_code, 200)
        other = create_app({"TESTING": True, "DATA_DIR": self.folder.name}).test_client()
        self.assertEqual(other.get("/api/settings").get_json()["maxScore"], 20)

    def test_m1_it_042_template_crud(self):
        body = {"title": "功能测试", "answerContent": "参考答案", "rubric": []}
        created = self.client.post("/api/templates", json=body)
        identifier = created.get_json()["id"]
        self.assertEqual(created.status_code, 201)
        updated = self.client.put("/api/templates/" + identifier,
                                  json={**body, "title": "已更新"})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(self.client.delete("/api/templates/" + identifier).status_code, 200)

    def test_m1_it_043_record_filter_and_pagination_validation(self):
        self.client.post("/compare_texts", json=self.payload)
        self.assertEqual(self.client.get("/api/records?q=不存在").get_json()["total"], 0)
        self.assertEqual(self.client.get("/api/records?status=passed").get_json()["total"], 1)
        self.assertEqual(self.client.get("/api/records?page=0").status_code, 400)

    def assert_rejected_without_record(self, expected=400, **changes):
        response = self.client.post("/compare_texts", json={**self.payload, **changes})
        self.assertEqual(response.status_code, expected, response.get_json())
        self.assertEqual(self.client.get("/api/stats").get_json()["total"], 0)

    def test_m1_it_010_giant_numeric_input_is_validation_error(self):
        self.assert_rejected_without_record(maxScore=10 ** 400)

    def test_m1_it_018_normalized_duplicate_rubric_is_rejected(self):
        self.assert_rejected_without_record(workContent="A", answerContent="AB", rubric=[
            {"keyword": "A", "weight": 1},
            {"keyword": "Ａ", "weight": 1},
            {"keyword": "Z", "weight": 1},
        ])

    def test_m1_it_035_invalid_transformer_probability_is_rejected(self):
        fake = types.ModuleType("homework.models.transformer")
        with patch.dict(sys.modules, {"homework.models.transformer": fake}):
            for value in (-0.1, 2.0, float("nan"), float("inf")):
                with self.subTest(model_output=repr(value)):
                    fake.calculate_similarity = lambda _work, _answer, output=value: output
                    response = self.client.post("/compare_texts", json={**self.payload,
                        "engine": "transformer", "workContent": "A", "answerContent": "AB",
                        "rubric": [{"keyword": "Z", "weight": 1}]})
                    self.assertEqual(response.status_code, 503, response.get_json())
        self.assertEqual(self.client.get("/api/stats").get_json()["total"], 0)

    def test_m1_it_032_giant_ai_response_keeps_local_score(self):
        cloud_content = json.dumps({"score": 10 ** 400})
        response_body = json.dumps({"choices": [{"message": {"content": cloud_content}}]})
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}), \
                patch("urllib.request.urlopen", return_value=io.BytesIO(response_body.encode("utf-8"))):
            response = self.client.post("/compare_texts", json={**self.payload,
                "workContent": "A", "answerContent": "AB", "useDeepseek": True, "aiWeight": 70})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["score"], 66.67)
        self.assertIsNone(response.get_json()["aiPercent"])
        self.assertTrue(response.get_json()["warnings"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
