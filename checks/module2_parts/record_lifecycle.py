"""Module 2 cases for record detail, summaries, deletion, and pagination."""
from common import ApiCase


class RecordLifecycleTests(ApiCase):
    def test_m2_it_005_record_detail_returns_full_audit_payload(self):
        created = self.create_record(rubric=[{"keyword": "发现缺陷", "weight": 2}])
        response = self.client.get(f"/api/records/{created['id']}")
        self.assertEqual(response.status_code, 200)
        for key in ("workContent", "answerContent", "rubric", "diff", "createdAt"):
            self.assertIn(key, response.get_json())

    def test_m2_it_006_record_summary_excludes_large_detail_fields(self):
        self.create_record()
        item = self.client.get("/api/records").get_json()["items"][0]
        for key in ("workContent", "answerContent", "rubric", "diff"):
            self.assertNotIn(key, item)

    def test_m2_it_007_delete_removes_record_and_updates_stats(self):
        created = self.create_record()
        response = self.client.delete(f"/api/records/{created['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/stats").get_json()["total"], 0)

    def test_m2_it_008_deleted_record_returns_404_on_second_access(self):
        created = self.create_record()
        self.client.delete(f"/api/records/{created['id']}")
        self.assertEqual(self.client.get(f"/api/records/{created['id']}").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/records/{created['id']}").status_code, 404)

    def test_m2_it_009_missing_record_export_returns_404(self):
        response = self.client.get("/api/records/not-found/export?format=json")
        self.assertEqual(response.status_code, 404)
        self.assertIn("不存在", response.get_json()["message"])

    def test_m2_it_010_second_page_is_stable_and_has_no_duplicates(self):
        for index in range(11):
            self.create_record(studentName=f"分页同学{index:02d}")
        first = self.client.get("/api/records?page=1&pageSize=10").get_json()
        second = self.client.get("/api/records?page=2&pageSize=10").get_json()
        first_ids = {item["id"] for item in first["items"]}
        second_ids = {item["id"] for item in second["items"]}
        self.assertEqual(len(first_ids), 10)
        self.assertEqual(len(second_ids), 1)
        self.assertFalse(first_ids & second_ids)

    def test_m2_it_011_missing_template_update_keeps_template_count(self):
        payload = {"title": "不存在", "answerContent": "参考答案", "rubric": [],
                   "options": BASE_OPTIONS}
        response = self.client.put("/api/templates/not-found", json=payload)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get("/api/stats").get_json()["templates"], 0)


BASE_OPTIONS = {"engine": "lexical", "maxScore": 100, "passPercent": 60,
                "aiWeight": 70, "useDeepseek": False}
