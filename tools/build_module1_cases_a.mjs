// Rebuild member A's case workbook from the supplied course template and real run evidence.
import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.resolve(import.meta.dirname, "..");
const source = "E:/Bocchi/QQReceiver/CourseMaterial/研究生课程/软件测试与质量保证实践/实践作业-文档模板2026/附录1：测试用例清单模板.xlsx";
const target = path.join(root, "deliverables/module1/曹浩-测试用例清单.xlsx");
const cases = JSON.parse(await fs.readFile(path.join(root, "tests/module1/cases_a.json"), "utf8"));
const run = JSON.parse(await fs.readFile(path.join(root, "reports/module1/a-final-full.json"), "utf8"));
const found = new Map(run.results.filter(row => row.case_id).map(row => [row.case_id, row]));
if (cases.length !== 22 || !run.successful || found.size !== cases.length ||
    cases.some(c => found.get(c.id)?.status !== "passed")) {
  throw new Error("Case catalog and real execution evidence do not match");
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const sheet = workbook.worksheets.getItem("Test Cases测试用例");
const info = workbook.worksheets.getItem("Information文档信息");
const values = cases.map(c => [
  c.id, c.item, c.title, c.id.startsWith("A-02") ? "高" : "中", "是", "新增",
  "基线2da8824；Python 3.12、Flask 3.1.3；独立临时数据库；外部依赖使用替身",
  c.input,
  `执行 tests/module1 中与 ${c.id} 对应的测试；检查响应及记录数；运行命令见 README`,
  c.expected,
  `断言符合预期；实际执行：${found.get(c.id).status}；证据 a-final-full.json`,
  "OK",
  `${c.method}；${c.requirement}${c.id >= "A-019" ? `；缺陷回归 A-DEF-${String(Number(c.id.slice(2)) - 18).padStart(3,"0")}` : ""}`,
]);
sheet.getRange("A2:M23").values = values;
sheet.getRange("A1:M1").format.rowHeight = 36;
sheet.getRange("A2:M23").format.rowHeight = 50;
sheet.getRange("A2:M23").format.wrapText = true;
sheet.getRange("A2:M23").format.verticalAlignment = "center";
const widths = {A:14,B:20,C:30,D:13,E:13,F:14,G:39,H:37,I:43,J:43,K:45,L:12,M:31};
for (const [col,width] of Object.entries(widths)) sheet.getRange(`${col}:${col}`).format.columnWidth = width;
sheet.freezePanes.freezeRows(1);
info.getRange("E7").values = [["2da8824"]];
info.getRange("J7").values = [["课程小组"]];
info.getRange("E8").values = [["阅知 · 成员A测试"]];
info.getRange("J8").values = [["M1-A"]];
info.getRange("E9").values = [["曹浩 M202677234"]];
info.getRange("J9").values = [["2026-09-15"]];
info.getRange("E10").values = [["待队友交叉复核"]];
info.getRange("J10").values = [["待复核"]];
info.getRange("E11").values = [["待审批"]];
info.getRange("J11").values = [["待审批"]];
info.getRange("B16").values = [["仅成员A的22条正式用例。小组总量待合并成员B用例；21条既有开发检查不计入新增用例。修复前4条失败、修复后22条通过。"]];
info.getRange("E13").formulas = [["=COUNTA('Test Cases测试用例'!$A$2:$A$1000)"]];
workbook.recalculate();
if (sheet.getRange("A23").values[0][0] !== "A-022" || info.getRange("E13").values[0][0] !== 22) {
  throw new Error("Workbook case count or final ID differs from source evidence");
}
await fs.mkdir(path.dirname(target), {recursive:true});
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(target);
console.log(`Wrote ${target}`);
