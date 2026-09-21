"""Module 2 cases for aggregate statistics and export contracts."""
import csv
import io

from common import ApiCase


class AnalyticsExportTests(ApiCase):
    def test_m2_it_012_empty_stats_use_null_averages_and_zero_bins(self):
        body = self.client.get("/api/stats").get_json()
        self.assertIsNone(body["average"])
        self.assertIsNone(body["passRate"])
        self.assertEqual(body["distribution"], [0, 0, 0, 0, 0])

    def test_m2_it_013_stats_use_expected_boundary_buckets(self):
        for similarity in (0, 19.99, 20, 39.99, 40, 59.99, 60, 79.99, 80, 100):
            self.store.save_records([{"similarity": similarity, "passed": similarity >= 60}])
        body = self.client.get("/api/stats").get_json()
        self.assertEqual(body["distribution"], [2, 2, 2, 2, 2])

    def test_m2_it_014_stats_average_pass_rate_and_template_count(self):
        self.store.save_records([{"similarity": 50, "passed": False},
                                 {"similarity": 100, "passed": True}])
        template = {"title": "统计模板", "answerContent": "答案", "rubric": [],
                    "options": {"engine": "lexical", "maxScore": 100, "passPercent": 60,
                                "aiWeight": 70, "useDeepseek": False}}
        self.assertEqual(self.client.post("/api/templates", json=template).status_code, 201)
        body = self.client.get("/api/stats").get_json()
        self.assertEqual((body["average"], body["passRate"], body["templates"]), (75.0, 50.0, 1))

    def test_m2_it_015_individual_json_export_preserves_unicode(self):
        created = self.create_record(studentName="张三")
        response = self.client.get(f"/api/records/{created['id']}/export?format=json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["Content-Disposition"])
        self.assertEqual(response.get_json()["studentName"], "张三")

    def test_m2_it_016_individual_csv_export_has_bom_header_and_one_row(self):
        created = self.create_record(studentName="李四")
        response = self.client.get(f"/api/records/{created['id']}/export?format=csv")
        text = response.data.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(rows[0][0], "记录编号")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][2], "李四")

    def test_m2_it_017_unsupported_individual_export_format_is_rejected(self):
        created = self.create_record()
        response = self.client.get(f"/api/records/{created['id']}/export?format=xml")
        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON 或 CSV", response.get_json()["message"])
