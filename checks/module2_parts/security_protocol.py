"""Module 2 cases for response headers and protocol-aware same-origin checks."""
from common import ApiCase, BASE_PAYLOAD


class SecurityProtocolTests(ApiCase):
    def test_m2_it_018_api_responses_disable_cache_and_framing(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["Referrer-Policy"], "same-origin")

    def test_m2_it_019_same_host_but_different_scheme_is_rejected(self):
        response = self.client.post("/compare_texts", json=BASE_PAYLOAD,
            headers={"Origin": "https://localhost"}, base_url="http://localhost")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/api/stats", base_url="http://localhost").get_json()["total"], 0)

    def test_m2_it_020_same_scheme_host_and_port_is_accepted(self):
        response = self.client.post("/compare_texts", json=BASE_PAYLOAD,
            headers={"Origin": "http://localhost"}, base_url="http://localhost")
        self.assertEqual(response.status_code, 200, response.get_json())
