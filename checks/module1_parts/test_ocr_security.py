"""模块一导出安全、同源保护与 OCR 文件测试（M1-IT-031～036）。"""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server import create_app


def png(color="white"):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, "PNG")
    stream.seek(0)
    return stream


class OcrSecurityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.client = create_app({"TESTING": True, "DATA_DIR": self.folder.name}).test_client()
        self.payload = {"studentName": "示例同学", "title": "模块一",
                        "workContent": "软件测试发现缺陷", "answerContent": "软件测试发现缺陷"}

    def tearDown(self):
        self.folder.cleanup()

    def test_m1_it_044_csv_formula_injection_is_escaped(self):
        self.client.post("/compare_texts", json={**self.payload, "studentName": "=1+1"})
        exported = self.client.get("/api/export").data.decode("utf-8-sig")
        self.assertIn("'=1+1", exported)

    def test_m1_it_045_print_report_escapes_html(self):
        created = self.client.post("/compare_texts", json={
            **self.payload, "workContent": "<script>alert(1)</script>"}).get_json()
        html = self.client.get("/records/" + created["id"] + "/print").data.decode()
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;", html)

    def test_m1_it_046_cross_origin_write_is_rejected(self):
        response = self.client.post("/compare_texts", json=self.payload,
                                    headers={"Origin": "https://example.com"})
        self.assertEqual(response.status_code, 403)

    def test_m1_it_047_ocr_rejects_missing_second_image(self):
        response = self.client.post("/ocr", data={"file1": (png(), "a.png")})
        self.assertEqual(response.status_code, 400)

    def test_m1_it_048_ocr_rejects_fake_image_before_engine_load(self):
        with patch("homework.ocr.recognize") as engine:
            response = self.client.post("/ocr", data={
                "file1": (io.BytesIO(b"bad"), "a.png"), "file2": (png(), "b.png")})
        self.assertEqual(response.status_code, 400)
        engine.assert_not_called()

    def test_m1_it_049_ocr_same_names_are_isolated_and_cleaned(self):
        observed = []

        def recognize(path, _engine, _language):
            with Image.open(path) as image:
                observed.append(image.getpixel((0, 0)))
            return "已识别文字"

        with patch("homework.ocr.status", return_value={"RapidOCR": True}), \
                patch("homework.ocr.recognize", side_effect=recognize):
            response = self.client.post("/ocr", data={
                "file1": (png("red"), "same.png"), "file2": (png("blue"), "same.png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(observed, [(255, 0, 0), (0, 0, 255)])
        self.assertEqual(list((Path(self.folder.name) / "temp").iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
