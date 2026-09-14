# Development verification

Date: 2026-09-14. Environment: Windows x64, Python 3.12.14, CPU.

## Verified

- `python -m unittest discover -s checks -v`: 21 developer regression checks passed. These check parameter validation, actual persistence and exports, atomic batch writes, isolated same-name uploads, image content validation and cleanup, AI failure fallback, safe score parsing, template/settings persistence, path safety, CSV/HTML escaping, Otsu preprocessing, classification metric denominators, local OCR path discovery, missing languages and incomplete model directories.
- `python -m pip check`: no broken requirements.
- JavaScript syntax check passed; all 25 project / research Python source files parsed successfully.
- Browser: real single and batch grading, template creation and application, history and record detail, desktop/mobile layouts.
- Browser: project demo images -> Otsu preprocessing -> RapidOCR -> editable recognized text -> local scoring -> saved record -> printable report.
- Original Paddle checkpoint loaded and used by both CLI and the web route. The model remains an experimental option because its scores are not reliable teaching grades.
- CLI image preprocessing and connected-component text extraction produced output successfully. Text comparison returned a computed result. ESRGAN CLI help runs without forcing optional PyTorch imports.
- Model weights, vocabulary, train/dev/test datasets match their original Git blob hashes after relocation.
- Source filenames outside workstation/runtime directories contain no non-ASCII names. No embedded API-key pattern remains in current source files; this does not remove old Git history.

## Limits

- No paid DeepSeek call was made. No API key is configured. Failure behavior is tested with a replacement adapter; real cloud responses require the owner's configuration.
- RapidOCR was tested on the included printed Chinese demo images; this is not a handwriting accuracy benchmark.
- The original Transformer returned approximately 0.180827 for an identical short-text pair. A diagnostic evaluation of the first 20 LCQMC test pairs yielded accuracy 0.55 and F1 0.40; these are small-sample observations, not a full benchmark.
- ESRGAN requires additional PyTorch and trustworthy weights and was not executed here.
- These are development checks for the project changes, not the course's formal test case spreadsheet, defect report, or final report.

The environment is recorded in `requirements/lock-windows-py312.txt`. Runtime records created during browser verification use explicit demo/example names and can be managed in the archive page.

## Local OCR installation and verification

- Installed PaddleOCR 3.2.0 and PaddleX 3.2.1, using existing PaddlePaddle 3.2.2 on CPU. Migrated the adapter to `predict()` and `rec_texts`.
- Downloaded PP-OCRv5 mobile detection/recognition models into `models/ocr/`. Archive hashes are pinned in `tools/setup_ocr_models.py`; each local directory includes source and file hashes in `download.json`.
- Installed the supplied Tesseract installer into `runtime/tesseract/`: version 5.5.3.20260724. Installed `chi_sim` and `eng` data. Fixed Windows language-path quoting by passing subprocess arguments separately, including paths with spaces.
- Actual `/ocr` -> lexical grading -> temporary record persistence passed for all three engines, each with printed Chinese and English samples (six flows). Identical recognized pairs scored 100; this checks the pipeline, not grading accuracy. Temporary upload directories were empty afterward.
- Observed PaddleOCR Chinese first-use pair took 15.45 seconds including initial import/loading; the subsequent English pair took 3.58 seconds on this machine. These are smoke-check timings, not a performance benchmark.
- Browser: Tesseract and PaddleOCR both recognized the two included demo images and populated editable work/answer fields. No extra grading records were created by these OCR browser checks.
- Rechecked original Transformer inference after installing OCR dependencies; inference still returns a computed result. `pip check` reports no broken requirements.
- OpenCV's standard and contrib distributions are both requested by upstream dependencies. Both are pinned to 4.10.0.84; `setup-paddleocr.bat` reinstalls contrib last because they share `cv2` files.
- Detailed local smoke output: `instance/ocr-smoke/results.json` (ignored runtime artifact). These printed samples do not establish handwriting, rotated-page, low-light, or real student-work accuracy.
