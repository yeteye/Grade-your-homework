// Rebuild the editable Member A presentation from versioned test evidence.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const root = path.resolve(import.meta.dirname, '..');
const skill = 'C:/Users/Bocchi/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const python = 'C:/Users/Bocchi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES = 'C:/Users/Bocchi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const stage = path.join(root, 'artifacts/dev-checks/module1/slides-a');
const output = path.join(root, 'deliverables/module1/曹浩-模块一演示.pptx');
await fs.mkdir(stage, { recursive: true });
await fs.mkdir(path.dirname(output), { recursive: true });
const { finalizePresentation, resolvePresentationFont } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont({ fontFamily: 'Microsoft YaHei' });
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const ink = '#182B3A';
const blue = '#23628A';
const muted = '#52636E';

function box(slide, value, x, y, w, h, size, color=ink, bold=false) {
  const shape = slide.shapes.add({
    geometry: 'textbox', position: { left:x, top:y, width:w, height:h },
    fill:'none', line:{ fill:'none', width:0 },
  });
  shape.text = value;
  shape.text.style = { typeface:font, fontSize:size, bold, color, autoFit:'none' };
  return shape;
}
function page(title, note) {
  const slide = presentation.slides.add();
  slide.background.fill = '#FFFFFF';
  box(slide, title, 68, 52, 1140, 78, 38, ink, true);
  box(slide, '软件测试与质量保证实践 · 模块一 · 曹浩 M202677234', 68, 661, 1130, 27, 16, muted);
  slide.speakerNotes.textFrame.setText(note);
  return slide;
}

{
  const s = presentation.slides.add();
  s.background.fill = '#FFFFFF';
  box(s, '评分与校验模块测试', 70, 210, 1130, 92, 58, ink, true);
  box(s, '曹浩  M202677234', 73, 329, 1080, 49, 30, blue);
  box(s, '被测基线 2da8824  ·  正式用例 22 条  ·  缺陷 4 项', 73, 392, 1100, 43, 23, muted);
  s.speakerNotes.textFrame.setText('本部分只说明曹浩从被测基线 2da8824 开始开展的测试工作，不涉及该提交之前的开发。');
}
{
  const s = page('测试对象与方法', '演示测试范围：输入校验、文本相似度评分、评分规则与 AI 响应兜底。运行命令见项目 README 和演示脚本。');
  box(s, '对象', 75, 166, 170, 42, 24, blue, true);
  box(s, '评分输入校验、文本相似度、评分规则、AI 响应兜底', 270, 166, 870, 70, 25);
  box(s, '方法', 75, 276, 170, 42, 24, blue, true);
  box(s, '等价类 6 条  ·  边界值 9 条  ·  场景法 7 条', 270, 276, 870, 70, 25);
  box(s, '环境', 75, 388, 170, 42, 24, blue, true);
  box(s, '隔离临时数据库与模型模拟响应；不调用付费接口', 270, 388, 890, 83, 25);
}
{
  const s = page('22 条正式用例', 'Excel 清单中的 A-001 至 A-022 与 tests/module1 中的自动化用例逐一对应；既有 checks/ 不计入新增课程用例。');
  box(s, 'A-001—A-006', 74, 158, 280, 40, 25, blue, true);
  box(s, '输入缺失、分值及权重边界', 360, 158, 780, 50, 25);
  box(s, 'A-007—A-012', 74, 263, 280, 40, 25, blue, true);
  box(s, '文本归一化与 Dice 相似度', 360, 263, 780, 50, 25);
  box(s, 'A-013—A-018', 74, 368, 280, 40, 25, blue, true);
  box(s, '评分规则、比例和兜底场景', 360, 368, 780, 50, 25);
  box(s, 'A-019—A-022', 74, 473, 280, 40, 25, blue, true);
  box(s, '基线缺陷复现与修复回归', 360, 473, 780, 50, 25);
}
{
  const s = page('基线复现的四项缺陷', '原始复现证据为 reports/module1/a-defects-baseline.json；各缺陷步骤、预期与实际结果见 docs/module1/defects/。');
  const rows = [
    ['A-DEF-001', '超大整数输入引发服务器错误'],
    ['A-DEF-002', '归一化后重复评分项被重复计分'],
    ['A-DEF-003', 'AI 分数按权重换算时使用错误基数'],
    ['A-DEF-004', 'AI 响应中的超大整数使兜底失败'],
  ];
  rows.forEach(([id, desc], i) => {
    const y = 155 + i*103;
    box(s, id, 75, y, 265, 42, 24, blue, true);
    box(s, desc, 350, y, 825, 73, 24);
  });
}
{
  const s = page('修复与回归结果', '本页统计来自版本化 JSON 执行记录。基线 18/22、修复后 22/22 均指曹浩新增正式用例；43/43 包括既有开发检查 21 条。');
  box(s, '被测基线 2da8824', 74, 156, 495, 45, 27, blue, true);
  box(s, '18 / 22 通过；4 条失败', 74, 214, 500, 61, 32, ink, true);
  box(s, '四项缺陷分别修复并提交', 74, 288, 500, 44, 24, muted);
  box(s, '最终回归', 678, 156, 475, 45, 27, blue, true);
  box(s, '22 / 22 正式用例通过', 678, 214, 530, 61, 32, ink, true);
  box(s, '连同既有检查共 43 / 43 通过', 678, 288, 530, 76, 24, muted);
  box(s, '证据：reports/module1/a-defects-baseline.json 与 a-final-full.json', 74, 510, 1100, 64, 21, muted);
}

const candidatePath = path.join(stage, 'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
for (let i=0; i<presentation.slides.items.length; i++) {
  const slide = presentation.slides.items[i];
  const preview = await presentation.export({ slide, format:'png', scale:1 });
  await fs.writeFile(path.join(stage, `slide-${i+1}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const result = await finalizePresentation({
  workspaceDir:root, candidatePath, finalPath:output,
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  explicitTotalSlideCount:5,
  requiredNativeTableOwnerSlides:[],
  requiredNativeChartOwnerSlides:[],
  fontPolicy:{ basis:'design', families:['Microsoft YaHei'] },
  verifyArtifactToolImport:true,
  receiptPath:path.join(stage, 'validation.json'),
});
console.log(JSON.stringify({ output, result }));
