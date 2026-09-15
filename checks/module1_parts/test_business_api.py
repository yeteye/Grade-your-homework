"""模块一批改、批量、设置、模板与记录接口测试（M1-IT-023～030）。"""
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_m1_it_023_compare_persists_record(self):
        created = self.client.post("/compare_texts", json=self.payload)
        self.assertEqual(created.status_code, 200)
        record = self.client.get("/api/records/" + created.get_json()["id"])
        self.assertEqual(record.get_json()["workContent"], self.payload["workContent"])

    def test_m1_it_024_invalid_compare_does_not_persist(self):
        response = self.client.post("/compare_texts", json={**self.payload, "workContent": " "})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/stats").get_json()["total"], 0)

    def test_m1_it_025_batch_accepts_one_and_twenty_items(self):
        for count in (1, 20):
            with self.subTest(count=count):
                data = {**self.payload, "useDeepseek": False,
                        "items": [{"studentName": f"学生{i}", "workContent": "正确"}
                                  for i in range(count)]}
                response = self.client.post("/api/batch", json=data)
                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.get_json()["count"], count)

    def test_m1_it_026_batch_rejects_zero_and_twenty_one_items(self):
        for count in (0, 21):
            with self.subTest(count=count):
                response = self.client.post("/api/batch", json={**self.payload, "items": [{}] * count})
                self.assertEqual(response.status_code, 400)

    def test_m1_it_027_batch_failure_is_atomic(self):
        data = {**self.payload, "items": [
            {"studentName": "甲", "workContent": "有效"},
            {"studentName": "乙", "workContent": ""}]}
        self.assertEqual(self.client.post("/api/batch", json=data).status_code, 400)
        self.assertEqual(self.client.get("/api/stats").get_json()["total"], 0)

    def test_m1_it_028_settings_persist_after_restart(self):
        self.assertEqual(self.client.put("/api/settings", json={"maxScore": 20}).status_code, 200)
        other = create_app({"TESTING": True, "DATA_DIR": self.folder.name}).test_client()
        self.assertEqual(other.get("/api/settings").get_json()["maxScore"], 20)

    def test_m1_it_029_template_crud(self):
        body = {"title": "功能测试", "answerContent": "参考答案", "rubric": []}
        created = self.client.post("/api/templates", json=body)
        identifier = created.get_json()["id"]
        self.assertEqual(created.status_code, 201)
        updated = self.client.put("/api/templates/" + identifier,
                                  json={**body, "title": "已更新"})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(self.client.delete("/api/templates/" + identifier).status_code, 200)

    def test_m1_it_030_record_filter_and_pagination_validation(self):
        self.client.post("/compare_texts", json=self.payload)
        self.assertEqual(self.client.get("/api/records?q=不存在").get_json()["total"], 0)
        self.assertEqual(self.client.get("/api/records?status=passed").get_json()["total"], 1)
        self.assertEqual(self.client.get("/api/records?page=0").status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
