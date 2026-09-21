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

