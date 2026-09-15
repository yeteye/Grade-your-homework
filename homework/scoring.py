"""Deterministic scoring and optional model orchestration, independent of Flask."""
import threading
from collections import Counter
from difflib import SequenceMatcher
from math import isfinite

from .ai_grader import get_points, GradingUnavailable
from .normalization import normalize
from .validation import text, options, rubric, ValidationError

_model_lock = threading.Lock()


def lexical_similarity(answer, reference):
    a, b = normalize(answer), normalize(reference)
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    return 2 * sum((ca & cb).values()) / (len(a) + len(b))


def compare(data, defaults):
    work = text(data.get('workContent'), '作业内容')
    answer = text(data.get('answerContent'), '参考答案')
    if not normalize(work) or not normalize(answer):
        raise ValidationError('作业和参考答案须包含文字或数字。')
    name = text(data.get('studentName', '未命名学生'), '学生姓名', 80)
    title = text(data.get('title', '未命名作业'), '作业名称', 120)
    config = options({**defaults, **{k: data[k] for k in defaults if k in data}})
    points = rubric(data.get('rubric', []))
    if any(not normalize(p['keyword']) for p in points):
        raise ValidationError('要点关键词须包含文字或数字。')
    matches = [{**p, 'matched': normalize(p['keyword']) in normalize(work)} for p in points]
    warnings = []
    if config['engine'] == 'transformer':
        if len(work) + len(answer) + 3 > 512:
            raise ValidationError('本地模型的作业与参考答案合计不能超过 509 字；请分题批改或切换文本基线。')
        try:
            with _model_lock:
                from .models.transformer import calculate_similarity
                base = float(calculate_similarity(work, answer))
        except (ImportError, OSError, RuntimeError, ValueError, OverflowError) as exc:
            raise GradingUnavailable('本地模型不可用，请检查依赖与权重，或切换文本基线。') from exc
        explanation = '实验 Transformer 返回文本匹配概率；原有权重尚未通过教学评分标定，不能作为正式成绩。'
    else:
        base = lexical_similarity(work, answer)
        explanation = '文本基线按规范化字符重合度计算；不能判断语义正误，请人工复核。'
    if not isfinite(base) or not 0 <= base <= 1:
        raise GradingUnavailable('评分引擎返回无效分数，未保存批改记录。')
    if points:
        coverage = sum(p['weight'] for p in matches if p['matched']) / sum(p['weight'] for p in matches)
        base = 0.4 * base + 0.6 * coverage
        explanation += ' 已启用要点：基础相似度占 40%，加权关键词覆盖占 60%。'
    ai_score = None
    final = base
    if config['useDeepseek'] and config['aiWeight'] > 0:
        try:
            ai_score = get_points(answer, work)
            ratio = config['aiWeight'] / 100
            final = base * (1 - ratio) + ai_score * ratio
        except GradingUnavailable as exc:
            warnings.append(str(exc))
    if not isfinite(final) or not 0 <= final <= 1:
        raise GradingUnavailable('评分引擎返回无效分数，未保存批改记录。')
    score = round(final * config['maxScore'], 2)
    similarity = round(final * 100, 2)
    passed = score >= round(config['maxScore'] * config['passPercent'] / 100, 2)
    diff = []
    if len(work) + len(answer) <= 2000:
        for tag, a1, a2, b1, b2 in SequenceMatcher(None, answer, work, autojunk=False).get_opcodes():
            diff.append({'type': tag, 'reference': answer[a1:a2], 'answer': work[b1:b2]})
    else:
        diff = [{'type': 'replace', 'reference': answer, 'answer': work}]
    return dict(studentName=name, title=title, workContent=work, answerContent=answer,
        options=config, rubric=matches, score=score, similarity=similarity,
        basePercent=round(base * 100, 2), aiPercent=None if ai_score is None else round(ai_score * 100, 2),
        passed=passed, explanation=explanation, warnings=warnings, diff=diff)
