"""模块一课程自动化测试：业务规则、接口、持久化与安全回归。"""
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from homework.ai_grader import GradingUnavailable, parse_score
from homework.preprocessing import prepare_image
from homework.scoring import compare, lexical_similarity, normalize
from homework.validation import ValidationError, number, options, rubric, text
from server import create_app


DEFAULTS = {
    "engine": "lexical",
    "maxScore": 100,
    "passPercent": 60,
    "aiWeight": 70,
    "useDeepseek": False,
}


def png(color="white"):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, "PNG")
    stream.seek(0)
    return stream


class ModuleOneValidationTests(unittest.TestCase):
    def test_m1_ut_001_text_accepts_boundary_length(self):
        self.assertEqual(text("甲" * 5000, "作业内容"), "甲" * 5000)

    def test_m1_ut_002_text_rejects_over_boundary(self):
        with self.assertRaises(ValidationError):
            text("甲" * 5001, "作业内容")

    def test_m1_ut_003_text_trims_surrounding_spaces(self):
        self.assertEqual(text("  软件测试  ", "作业内容"), "软件测试")

    def test_m1_ut_004_required_text_rejects_blank(self):
        with self.assertRaises(ValidationError):
            text(" \n\t ", "作业内容")

    def test_m1_ut_005_number_accepts_minimum_and_maximum(self):
        self.assertEqual(number(1, "满分", 1, 1000), 1.0)
        self.assertEqual(number(1000, "满分", 1, 1000), 1000.0)

    def test_m1_ut_006_number_rejects_outside_range(self):
        for value in (0, 1001):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                number(value, "满分", 1, 1000)

    def test_m1_ut_007_number_rejects_boolean(self):
        with self.assertRaises(ValidationError):
            number(True, "满分", 1, 1000)

    def test_m1_ut_008_options_accept_valid_boundaries(self):
        value = options({"engine": "lexical", "maxScore": 1, "passPercent": 0,
                         "aiWeight": 100, "useDeepseek": False})
        self.assertEqual((value["maxScore"], value["passPercent"], value["aiWeight"]), (1, 0, 100))

    def test_m1_ut_009_options_reject_unknown_engine(self):
        with self.assertRaises(ValidationError):
            options({"engine": "unknown"})

    def test_m1_ut_010_rubric_accepts_twenty_items(self):
        items = [{"keyword": f"要点{i}", "weight": 1} for i in range(20)]
        self.assertEqual(len(rubric(items)), 20)

    def test_m1_ut_011_rubric_rejects_twenty_one_items(self):
        items = [{"keyword": f"要点{i}", "weight": 1} for i in range(21)]
        with self.assertRaises(ValidationError):
            rubric(items)

    def test_m1_ut_012_rubric_rejects_duplicate_keyword(self):
        with self.assertRaises(ValidationError):
            rubric([{"keyword": "缺陷", "weight": 1}, {"keyword": "缺陷", "weight": 2}])


class ModuleOneScoringTests(unittest.TestCase):
    def payload(self, **changes):
        value = {"studentName": "测试同学", "title": "模块一", "workContent": "软件测试发现缺陷",
                 "answerContent": "软件测试发现缺陷"}
        value.update(changes)
        return value

    def test_m1_ut_013_normalize_unifies_width_case_and_punctuation(self):
        self.assertEqual(normalize("ＡbＣ！"), "abc")

    def test_m1_ut_014_lexical_identical_text_scores_one(self):
        self.assertEqual(lexical_similarity("软件测试", "软件测试"), 1)

    def test_m1_ut_015_lexical_empty_normalized_text_scores_zero(self):
        self.assertEqual(lexical_similarity("!!!", "软件测试"), 0)

    def test_m1_ut_016_weighted_rubric_changes_score(self):
        result = compare(self.payload(workContent="软件测试", rubric=[
            {"keyword": "软件", "weight": 1}, {"keyword": "缺陷", "weight": 1}]), DEFAULTS)
        expected = 0.4 * lexical_similarity("软件测试", "软件测试发现缺陷") + 0.6 * 0.5
        self.assertAlmostEqual(result["similarity"], round(expected * 100, 2))

    def test_m1_ut_017_displayed_score_controls_pass_boundary(self):
        result = compare(self.payload(maxScore=10, passPercent=100), DEFAULTS)
        self.assertEqual(result["score"], 10)
        self.assertTrue(result["passed"])

    def test_m1_ut_018_ai_failure_preserves_base_score(self):
        with patch("homework.scoring.get_points", side_effect=GradingUnavailable("模拟超时")):
            result = compare(self.payload(useDeepseek=True), DEFAULTS)
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["warnings"], ["模拟超时"])

    def test_m1_ut_019_ai_valid_score_is_weighted(self):
        with patch("homework.scoring.get_points", return_value=0.5):
            result = compare(self.payload(useDeepseek=True, aiWeight=40), DEFAULTS)
        self.assertEqual(result["score"], 80)

    def test_m1_ut_020_ai_response_rejects_invalid_values(self):
        for value in ("-0.1", "1.1", "NaN", "true", '{"score":"0.5"}'):
            with self.subTest(value=value), self.assertRaises(GradingUnavailable):
                parse_score(value)

    def test_m1_ut_021_transformer_accepts_combined_length_509(self):
        fake = types.ModuleType("homework.models.transformer")
        fake.calculate_similarity = lambda _work, _answer: 0.8
        with patch.dict(sys.modules, {"homework.models.transformer": fake}):
            result = compare(self.payload(workContent="甲" * 254, answerContent="乙" * 255,
                                          engine="transformer"), DEFAULTS)
        self.assertEqual(result["score"], 80)

    def test_m1_ut_022_transformer_rejects_combined_length_510(self):
        with self.assertRaises(ValidationError):
            compare(self.payload(workContent="甲" * 255, answerContent="乙" * 255,
                                 engine="transformer"), DEFAULTS)


class ModuleOneApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.client = create_app({"TESTING": True, "DATA_DIR": self.folder.name}).test_client()
        self.payload = {"studentName": "示例同学", "title": "模块一", "workContent": "软件测试发现缺陷",
                        "answerContent": "软件测试发现缺陷"}

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
                        "items": [{"studentName": f"学生{i}", "workContent": "正确"} for i in range(count)]}
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
            {"studentName": "甲", "workContent": "有效"}, {"studentName": "乙", "workContent": ""}]}
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
        self.assertEqual(self.client.put("/api/templates/" + identifier, json={**body, "title": "已更新"}).status_code, 200)
        self.assertEqual(self.client.delete("/api/templates/" + identifier).status_code, 200)

    def test_m1_it_030_record_filter_and_pagination_validation(self):
        self.client.post("/compare_texts", json=self.payload)
        self.assertEqual(self.client.get("/api/records?q=不存在").get_json()["total"], 0)
        self.assertEqual(self.client.get("/api/records?status=passed").get_json()["total"], 1)
        self.assertEqual(self.client.get("/api/records?page=0").status_code, 400)

    def test_m1_it_031_csv_formula_injection_is_escaped(self):
        self.client.post("/compare_texts", json={**self.payload, "studentName": "=1+1"})
        exported = self.client.get("/api/export").data.decode("utf-8-sig")
        self.assertIn("'=1+1", exported)

    def test_m1_it_032_print_report_escapes_html(self):
        created = self.client.post("/compare_texts", json={**self.payload,
            "workContent": "<script>alert(1)</script>"}).get_json()
        html = self.client.get("/records/" + created["id"] + "/print").data.decode()
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;", html)

    def test_m1_it_033_cross_origin_write_is_rejected(self):
        response = self.client.post("/compare_texts", json=self.payload,
                                    headers={"Origin": "https://example.com"})
        self.assertEqual(response.status_code, 403)

    def test_m1_it_034_ocr_rejects_missing_second_image(self):
        response = self.client.post("/ocr", data={"file1": (png(), "a.png")})
        self.assertEqual(response.status_code, 400)

    def test_m1_it_035_ocr_rejects_fake_image_before_engine_load(self):
        with patch("homework.ocr.recognize") as engine:
            response = self.client.post("/ocr", data={"file1": (io.BytesIO(b"bad"), "a.png"),
                                                        "file2": (png(), "b.png")})
        self.assertEqual(response.status_code, 400)
        engine.assert_not_called()

    def test_m1_it_036_ocr_same_names_are_isolated_and_cleaned(self):
        observed = []

        def recognize(path, _engine, _language):
            with Image.open(path) as image:
                observed.append(image.getpixel((0, 0)))
            return "已识别文字"

        with patch("homework.ocr.status", return_value={"RapidOCR": True}), \
                patch("homework.ocr.recognize", side_effect=recognize):
            response = self.client.post("/ocr", data={"file1": (png("red"), "same.png"),
                                                        "file2": (png("blue"), "same.png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(observed, [(255, 0, 0), (0, 0, 255)])
        self.assertEqual(list((Path(self.folder.name) / "temp").iterdir()), [])


class ModuleOneImageTests(unittest.TestCase):
    def test_m1_ut_037_grayscale_mode_outputs_rgb(self):
        result = prepare_image(Image.new("RGB", (2, 2), "red"), "grayscale")
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0))[0], result.getpixel((0, 0))[1])

    def test_m1_ut_038_binary_mode_preserves_dark_text(self):
        image = Image.new("RGB", (4, 2), "white")
        image.putpixel((0, 0), (20, 20, 20))
        result = prepare_image(image, "binary")
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((3, 1)), (255, 255, 255))

    def test_m1_ut_039_unknown_preprocessing_mode_is_rejected(self):
        with self.assertRaises(ValidationError):
            prepare_image(Image.new("RGB", (2, 2), "white"), "unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
