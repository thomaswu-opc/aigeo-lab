// 回归：控制台的摘要函数不能被一种「合法但少字段」的工具返回打断。
//
// 背景：select 对没有三向尺寸的实体会返回 "dims": [null,null,null]。
// digestOf 里原来直接 x.toFixed，一个 null 就抛 TypeError —— renderFeed 中断，
// loadRun 后面的 renderRight / renderStatus 全没执行，页面停在「未载入」，
// 几何画面只剩一个 <img alt="scene"> 的破图图标。看起来像「没有画面」，实际是渲染崩了。
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "studio.html"), "utf8");

// 从页面里抠出 digestOf（保持「测的就是线上那份代码」，不复制一份实现）
const start = html.indexOf("function digestOf(s){");
if (start < 0) throw new Error("studio.html 里找不到 digestOf");
let depth = 0, end = start;
for (let i = html.indexOf("{", start); i < html.length; i++) {
  if (html[i] === "{") depth++;
  else if (html[i] === "}") { depth--; if (depth === 0) { end = i + 1; break; } }
}
const src = html.slice(start, end);
const digestOf = new Function("s", src.replace(/^function digestOf\(s\)\{/, "") .replace(/\}$/, ""));

const cases = [
  ["select 里 dims 为 null", { items: [{ sid: "C000", dims: [null, null, null] }] }],
  ["select 里没有 dims 也没有 d0", { items: [{ sid: "C001" }] }],
  ["rows 里 value 缺失", { rows: [{ sid: "C002" }] }],
  ["全是 null 的数值字段", { value: null, witness_a: [0, 0, 0], witness_b: [0, 0, 0] }],
  ["空对象", {}],
  ["正常的 select", { items: [{ sid: "C017", dims: [0.5, 0.5, 0.9] }] }],
];

let bad = 0;
for (const [name, payload] of cases) {
  try {
    const out = digestOf(JSON.stringify(payload));
    if (typeof out !== "string") throw new Error("返回的不是字符串");
    console.log(`ok   ${name} → ${out.slice(0, 60)}`);
  } catch (e) {
    bad++;
    console.log(`FAIL ${name} → ${e.message}`);
  }
}
console.log(bad ? `\n${bad} 项失败` : "\n全部通过");
process.exit(bad ? 1 : 0);
