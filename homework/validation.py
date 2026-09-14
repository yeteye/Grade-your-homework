import math


class ValidationError(ValueError):
    pass


def text(value, label, limit=5000, required=True):
    if not isinstance(value, str):
        raise ValidationError(f'{label}必须是文本。')
    value = value.strip()
    if required and not value:
        raise ValidationError(f'请填写{label}。')
    if len(value) > limit:
        raise ValidationError(f'{label}不能超过 {limit} 个字符。')
    return value


def number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f'{label}必须是数字。')
    # Check the bounded interval first: isfinite() converts integers to float
    # and raises OverflowError for arbitrarily large JSON integers.
    if not low <= value <= high or not math.isfinite(value):
        raise ValidationError(f'{label}须在 {low}～{high} 之间。')
    return float(value)


def options(data):
    if not isinstance(data, dict):
        raise ValidationError('评分设置必须是对象。')
    engine = data.get('engine', 'lexical')
    if engine not in ('lexical', 'transformer'):
        raise ValidationError('不支持的评分引擎。')
    ai = data.get('useDeepseek', False)
    if not isinstance(ai, bool):
        raise ValidationError('AI 开关必须为布尔值。')
    return {'engine': engine,
        'maxScore': number(data.get('maxScore', 100), '满分', 1, 1000),
        'passPercent': number(data.get('passPercent', 60), '达标线', 0, 100),
        'aiWeight': number(data.get('aiWeight', 70), 'AI 权重', 0, 100), 'useDeepseek': ai}


def rubric(value):
    if not isinstance(value, list) or len(value) > 20:
        raise ValidationError('评分要点应为列表，最多 20 项。')
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise ValidationError('评分要点格式错误。')
        result.append({'keyword': text(item.get('keyword'), '要点关键词', 80),
                       'weight': number(item.get('weight', 1), '要点权重', 0.1, 100)})
    if len({r['keyword'].casefold() for r in result}) != len(result):
        raise ValidationError('评分要点关键词不能重复。')
    return result
