"""Run course/developer tests and save versioned, machine-readable evidence."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def git(*args, binary=False):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=not binary).strip()


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def startTest(self, test):
        super().startTest(test)
        self.started = time.perf_counter()
        match = re.search(r'test_a(\d{3})', test.id())
        self.row = {'test': test.id(), 'case_id': f'A-{match[1]}' if match else None,
                    'status': 'passed', 'subtests': []}

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.row.update(status='failed', detail=self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self.row.update(status='error', detail=self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.row.update(status='skipped', detail=reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.row.update(status='expected_failure', detail=self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.row['status'] = 'unexpected_success'

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        child = {'parameters': str(subtest.params), 'status': 'passed' if err is None else 'failed'}
        if err:
            child['detail'] = self._exc_info_to_string(err, test)
            self.row['status'] = 'failed'
        self.row['subtests'].append(child)

    def stopTest(self, test):
        self.row['seconds'] = round(time.perf_counter() - self.started, 6)
        self.rows.append(self.row)
        super().stopTest(test)


def hashes(base, paths):
    return {str(p.relative_to(base)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in paths for p in sorted((base / folder).rglob('*.py')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['all', 'course', 'checks'], default='all')
    parser.add_argument('--label', default='latest')
    parser.add_argument('--source-ref', help='Run current tests against source archived from a Git commit')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/module1')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label):
        parser.error('label must contain ASCII letters, digits, underscore or hyphen')
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / (args.label + '.json')
    text_path = args.output / (args.label + '.txt')
    if args.label != 'latest' and (json_path.exists() or text_path.exists()):
        parser.error('Named evidence already exists; choose a new label to preserve prior runs')
    with tempfile.TemporaryDirectory(prefix='module1-run-') as folder:
        temporary = Path(folder)
        source = ROOT
        if args.source_ref:
            source = temporary / 'source'
            source.mkdir()
            archive = subprocess.check_output(['git', 'archive', '--format=zip', args.source_ref], cwd=ROOT)
            with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
                for member in zipped.infolist():
                    destination = (source / member.filename).resolve()
                    if not destination.is_relative_to(source.resolve()):
                        raise ValueError('Unsafe archive member')
                zipped.extractall(source)
        os.environ['HOMEWORK_DATA_DIR'] = str(temporary / 'app-data')
        os.environ.pop('DEEPSEEK_API_KEY', None)
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(source))
        suite = unittest.TestSuite()
        if args.suite in ('all', 'checks'):
            suite.addTests(unittest.TestLoader().discover(str(ROOT / 'checks')))
        if args.suite in ('all', 'course'):
            suite.addTests(unittest.TestLoader().discover(str(ROOT / 'tests/module1'), top_level_dir=str(ROOT)))
        buffer = io.StringIO()
        started = datetime.now(timezone.utc).isoformat()
        result = unittest.TextTestRunner(stream=buffer, verbosity=2, resultclass=EvidenceResult).run(suite)
        versions = {}
        for package in ('Flask', 'Pillow', 'Werkzeug', 'waitress'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = 'not installed'
        statuses = {s: sum(r['status'] == s for r in result.rows)
                    for s in ('passed', 'failed', 'error', 'skipped', 'expected_failure', 'unexpected_success')}
        sources = hashes(source, ['homework'])
        sources['server.py'] = hashlib.sha256((source / 'server.py').read_bytes()).hexdigest()
        evidence = dict(started_at_utc=started, source_commit=git('rev-parse', args.source_ref or 'HEAD'),
            source_mode='git archive' if args.source_ref else 'working tree', test_commit=git('rev-parse', 'HEAD'),
            source_hashes=sources, test_hashes=hashes(ROOT, ['tests/module1', 'checks']),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            python=sys.version, platform=platform.platform(), dependencies=versions,
            command=sys.argv, suite=args.suite, tests_run=result.testsRun, outcomes=statuses,
            formal_cases=sum(r['case_id'] is not None for r in result.rows),
            parameter_variants=sum(len(r['subtests']) for r in result.rows),
            successful=result.wasSuccessful(), results=result.rows)
        json_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
        text_path.write_text(buffer.getvalue(), encoding='utf-8')
        print(buffer.getvalue())
        print(f'Evidence: {json_path}')
        return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
