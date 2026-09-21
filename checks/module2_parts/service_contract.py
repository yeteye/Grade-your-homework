"""Module 2 cases for health, demo resources, and HTTP error contracts."""
from common import ApiCase


class ServiceContractTests(ApiCase):
    def test_m2_it_001_health_reports_runtime_contract(self):
        response = self.client.get("/api/health")
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["limits"], {"textLength": 5000, "imageMB": 8, "batchSize": 20})
        self.assertEqual(body["transformer"]["maxCombinedLength"], 509)

    def test_m2_it_002_demo_images_are_real_png_files(self):
        for kind in ("work", "answer"):
            response = self.client.get(f"/api/demo-images/{kind}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue(response.mimetype.startswith("image/"))
            response.close()

    def test_m2_it_003_unknown_demo_image_returns_json_404(self):
        response = self.client.get("/api/demo-images/unknown")
        self.assertEqual(response.status_code, 404)
        self.assertIn("不存在", response.get_json()["message"])

    def test_m2_it_004_unsupported_method_returns_json_405(self):
        response = self.client.post("/api/stats", json={})
        self.assertEqual(response.status_code, 405)
        self.assertIn("不支持", response.get_json()["message"])
