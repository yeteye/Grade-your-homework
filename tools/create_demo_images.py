"""Generate readable, synthetic input images for the offline OCR demonstration."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
destination = root / 'examples'
destination.mkdir(exist_ok=True)
font_path = Path('C:/Windows/Fonts/msyh.ttc')
font = ImageFont.truetype(str(font_path), 32)
heading = ImageFont.truetype(str(font_path), 38)
samples = {
    'work': ['软件测试的目的是发现软件缺陷，', '验证软件是否满足需求。', '测试通过运行程序和检查结果发现问题。'],
    'answer': ['软件测试的目的是发现软件缺陷，', '验证软件是否满足需求，', '并为软件质量评估提供依据。'],
}
for kind, lines in samples.items():
    image = Image.new('RGB', (1100, 540), '#ffffff')
    draw = ImageDraw.Draw(image)
    draw.text((70, 60), '软件测试基础', font=heading, fill='#253d35')
    draw.line((70, 130, 1030, 130), fill='#a5b7aa', width=2)
    for index, line in enumerate(lines):
        draw.text((70, 190 + index * 75), line, font=font, fill='#18241e')
    image.save(destination / f'demo-{kind}.png')
print('Created examples/demo-work.png and examples/demo-answer.png')
