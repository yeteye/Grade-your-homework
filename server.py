"""Local homework review app. Run with `python server.py`."""
import csv
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, render_template, Response
from werkzeug.exceptions import HTTPException
from homework.ai_grader import GradingUnavailable
from homework import ocr
from homework.scoring import compare, normalize
from homework.storage import Store
from homework.validation import ValidationError, text, options, rubric

ROOT = Path(__file__).resolve().parent


def create_app(config=None):
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config.update(MAX_CONTENT_LENGTH=17 * 1024 * 1024, MAX_FORM_MEMORY_SIZE=128 * 1024,
                      TRUSTED_HOSTS=['127.0.0.1', 'localhost', '[::1]'],
                      DATA_DIR=os.environ.get('HOMEWORK_DATA_DIR', str(ROOT / 'instance')))
    if config:
        app.config.update(config)
    data_dir = Path(app.config['DATA_DIR'])
    data_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = data_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    store = Store(data_dir / 'homework.sqlite3')
    app.extensions['store'] = store

    @app.before_request
    def same_origin():
        if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
            origin = request.headers.get('Origin')
            if origin and urlsplit(origin).netloc != request.host:
                return jsonify(message='不允许跨站修改数据。'), 403

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        if request.path.startswith('/api') or request.path in ('/ocr', '/compare_texts'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(ValidationError)
    def invalid(exc):
        return jsonify(message=str(exc)), 400

    @app.errorhandler(GradingUnavailable)
    def unavailable(exc):
        return jsonify(message=str(exc)), 503

    @app.errorhandler(HTTPException)
    def http_error(exc):
        messages = {400: '请求格式错误。', 404: '记录或页面不存在。', 405: '请求方法不支持。',
                    413: '上传总量过大，单张图片限制 8 MB。', 415: '请使用 JSON 格式提交数据。'}
        return jsonify(message=messages.get(exc.code, exc.description)), exc.code

    @app.errorhandler(Exception)
    def unexpected(exc):
        app.logger.exception('Request failed')
        return jsonify(message='服务暂时无法完成请求，请稍后重试。'), 500

    def body():
        value = request.get_json()
        if not isinstance(value, dict):
            raise ValidationError('请求正文必须是 JSON 对象。')
        return value

    def record_or_404(record_id):
        from flask import abort
        result = store.record(record_id)
        if result is None:
            abort(404)
        return result

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/api/health')
    def health():
        model_path = Path(os.environ.get('HOMEWORK_MODEL_PATH', ROOT / 'models/transformer/model_best.pdparams'))
        return jsonify(status='ok', app='homework-studio', ocr=ocr.status(), ocrDetails=ocr.details(),
            transformer={'installed': importlib.util.find_spec('paddle') is not None,
                         'weights': model_path.is_file(), 'maxCombinedLength': 509},
            deepseek=bool(os.environ.get('DEEPSEEK_API_KEY', '').strip()),
            limits={'textLength': 5000, 'imageMB': 8, 'batchSize': 20},
            note='依赖状态不代表模型已加载；首次运行会执行真实推理。')

    @app.get('/api/demo-images/<kind>')
    def demo_image(kind):
        from flask import abort, send_file
        if kind not in ('work', 'answer'):
            abort(404)
        return send_file(ROOT / 'examples' / f'demo-{kind}.png')

    @app.route('/api/settings', methods=['GET', 'PUT'])
    def settings():
        if request.method == 'GET':
            return jsonify(store.settings())
        return jsonify(store.save_settings(options(body())))

    @app.post('/ocr')
    def recognize():
        if 'file1' not in request.files or 'file2' not in request.files:
            raise ValidationError('请同时提供作业图片和参考答案图片。')
        return jsonify(ocr.recognize_pair(request.files['file1'], request.files['file2'],
            request.form.get('model', 'RapidOCR'), request.form.get('language', '中文'), temp_dir,
            request.form.get('preprocessing', 'original')))

    @app.post('/compare_texts')
    def compare_texts():
        result = compare(body(), store.settings())
        store.save_records([result])
        return jsonify(result)

    @app.post('/api/batch')
    def batch():
        data = body()
        items = data.get('items')
        if not isinstance(items, list) or not 1 <= len(items) <= 20:
            raise ValidationError('批量批改每次须包含 1～20 份作业。')
        if data.get('useDeepseek') or store.settings()['useDeepseek']:
            # Explicit per-batch false may override an AI-enabled saved setting.
            if data.get('useDeepseek') is not False:
                raise ValidationError('批量模式请关闭 AI 增强，避免长时间等待。')
        results = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValidationError(f'第 {i + 1} 份作业格式错误。')
            try:
                results.append(compare({**data, 'studentName': item.get('studentName'),
                    'workContent': item.get('workContent')}, store.settings()))
            except ValidationError as exc:
                raise ValidationError(f'第 {i + 1} 份：{exc}') from None
        store.save_records(results)
        return jsonify(items=results, count=len(results)), 201

    def filtered_records():
        q = request.args.get('q', '').strip().casefold()
        status = request.args.get('status', 'all')
        if status not in ('all', 'passed', 'review'):
            raise ValidationError('无效的记录筛选条件。')
        return [r for r in store.records()
                if (not q or q in (r['studentName'] + ' ' + r['title']).casefold())
                and (status == 'all' or r['passed'] == (status == 'passed'))]

    @app.get('/api/records')
    def records():
        try:
            page = int(request.args.get('page', 1))
            size = int(request.args.get('pageSize', 10))
        except ValueError:
            raise ValidationError('页码必须为整数。') from None
        if page < 1 or not 1 <= size <= 100:
            raise ValidationError('页码或每页数量超出范围。')
        items = filtered_records()
        summaries = [{k: v for k, v in r.items() if k not in ('diff', 'workContent', 'answerContent', 'rubric')}
                     for r in items[(page - 1) * size:page * size]]
        return jsonify(items=summaries, total=len(items), page=page, pageSize=size)

    @app.route('/api/records/<record_id>', methods=['GET', 'DELETE'])
    def record_detail(record_id):
        result = record_or_404(record_id)
        if request.method == 'DELETE':
            store.delete_record(record_id)
            return jsonify(message='批改记录已删除。')
        return jsonify(result)

    @app.get('/api/stats')
    def stats():
        items = store.records()
        count = len(items)
        bins = [0] * 5
        for item in items:
            bins[min(4, int(item['similarity'] // 20))] += 1
        return jsonify(total=count,
            average=round(sum(r['similarity'] for r in items) / count, 1) if count else None,
            passRate=round(100 * sum(r['passed'] for r in items) / count, 1) if count else None,
            templates=len(store.templates()), distribution=bins)

    def csv_response(items):
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['记录编号', '批改时间', '学生', '作业', '得分', '满分', '达标', '引擎', '作业内容', '参考答案'])
        def safe(value):
            s = str(value)
            return "'" + s if s.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else s
        for r in items:
            writer.writerow([safe(v) for v in (r['id'], r['createdAt'], r['studentName'], r['title'],
                r['score'], r['options']['maxScore'], '是' if r['passed'] else '否', r['options']['engine'],
                r['workContent'], r['answerContent'])])
        return Response('\ufeff' + stream.getvalue(), mimetype='text/csv',
                        headers={'Content-Disposition': 'attachment; filename="homework-records.csv"'})

    @app.get('/api/export')
    def export_all():
        return csv_response(filtered_records())

    @app.get('/api/records/<record_id>/export')
    def export_one(record_id):
        value = record_or_404(record_id)
        fmt = request.args.get('format', 'json')
        if fmt == 'csv':
            return csv_response([value])
        if fmt == 'json':
            return Response(json.dumps(value, ensure_ascii=False, indent=2), mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename="homework-{record_id[:8]}.json"'})
        raise ValidationError('导出格式仅支持 JSON 或 CSV。')

    @app.get('/records/<record_id>/print')
    def print_record(record_id):
        from datetime import datetime
        record = record_or_404(record_id)
        display_time = datetime.fromisoformat(record['createdAt']).astimezone().strftime('%Y-%m-%d %H:%M')
        return render_template('report.html', record=record, display_time=display_time)

    def template_value(data):
        answer = text(data.get('answerContent'), '参考答案')
        if not normalize(answer):
            raise ValidationError('参考答案须包含文字或数字。')
        points = rubric(data.get('rubric', []))
        if any(not normalize(p['keyword']) for p in points):
            raise ValidationError('要点关键词须包含文字或数字。')
        return {'title': text(data.get('title'), '模板名称', 120), 'answerContent': answer,
                'rubric': points, 'options': options(data.get('options', store.settings()))}

    @app.route('/api/templates', methods=['GET', 'POST'])
    def templates():
        if request.method == 'GET':
            return jsonify(items=store.templates())
        return jsonify(store.save_template(template_value(body()))), 201

    @app.route('/api/templates/<template_id>', methods=['PUT', 'DELETE'])
    def template_detail(template_id):
        from flask import abort
        if request.method == 'DELETE':
            if not store.delete_template(template_id):
                abort(404)
            return jsonify(message='模板已删除。')
        result = store.save_template(template_value(body()), template_id)
        if result is None:
            abort(404)
        return jsonify(result)

    return app


app = create_app()


if __name__ == '__main__':
    from waitress import serve
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get('PORT', '5000'))
    print(f'Homework Studio: http://127.0.0.1:{port}', flush=True)
    serve(app, host='127.0.0.1', port=port, threads=4)
