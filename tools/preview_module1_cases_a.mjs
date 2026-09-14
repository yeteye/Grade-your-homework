// Internal visual QA for the final A workbook; previews are ignored by Git.
import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
const root = path.resolve(import.meta.dirname, "..");
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root, "deliverables/module1/曹浩-测试用例清单.xlsx")));
const folder = path.join(root, "artifacts/dev-checks/module1");
await fs.mkdir(folder, {recursive:true});
for (const [name,sheetName,range] of [
  ["cases-left", "Test Cases测试用例", "A1:F7"],
  ["cases-right", "Test Cases测试用例", "G1:M7"],
  ["cases-last", "Test Cases测试用例", "H18:M23"],
  ["info", "Information文档信息", "B6:K16"],
]) {
  const image = await workbook.render({sheetName,range,scale:1,format:"png"});
  const file = path.join(folder,`${name}.png`);
  await fs.writeFile(file,new Uint8Array(await image.arrayBuffer()));
  console.log(file);
}
