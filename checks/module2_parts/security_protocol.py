"""Module 2 cases for response headers and protocol-aware same-origin checks."""
from common import ApiCase, BASE_PAYLOAD


class SecurityProtocolTests(ApiCase):
    def test_m2_it_018_api_responses_disable_cache_and_framing(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["Referrer-Policy"], "same-origin")

