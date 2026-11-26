// 用 CDP 截「元素截图」：整页截图里右栏只有 280 px 宽，看不清三维画得对不对。
const PORT = +(process.env.CDP_PORT || 9223);
const raw = process.argv[2];
const sel = process.argv[3] || "#viewwrap";
const out = process.argv[4] || "snap.png";
const [base0, hash0] = raw.split("#");
const URL_ = base0 + (base0.includes("?") ? "&" : "?") + "cb=" + Date.now() + (hash0 ? "#" + hash0 : "");
const tabs = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const page = tabs.find(t => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
const send = (m, p = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method: m, params: p })); });
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result || m.error); pending.delete(m.id); } };
await new Promise(r => ws.onopen = r);
await send("Page.enable");
await send("Runtime.enable");
// 放大右栏再截图：把右栏宽度临时拉到 620 px，看的是真实渲染
const W = +(process.env.W || 1680);
await send("Emulation.setDeviceMetricsOverride", { width: W, height: 1050, deviceScaleFactor: +(process.env.DPR || 2), mobile: false });
await send("Page.navigate", { url: URL_ });
await new Promise(r => setTimeout(r, 8000));
await send("Runtime.evaluate", { expression: `(()=>{const g=document.querySelector('.lab'); if(g){g.style.gridTemplateColumns='200px minmax(0,1fr) 620px';} window.dispatchEvent(new Event('resize')); return 1})()` });
await new Promise(r => setTimeout(r, 2500));
if (process.env.CAM) {
  await send("Runtime.evaluate", { expression: process.env.CAM });
  await new Promise(r => setTimeout(r, 1200));
}
const box = await send("Runtime.evaluate", { expression: `(()=>{const e=document.querySelector('${sel}'); if(!e) return null; const b=e.getBoundingClientRect(); return {x:b.x,y:b.y,w:b.width,h:b.height}})()`, returnByValue: true });
const b = box.result.value;
if (!b) { console.log("元素不存在:", sel); process.exit(1); }
const shot = await send("Page.captureScreenshot", { format: "png", clip: { x: b.x, y: b.y, width: b.w, height: b.h, scale: 2 } });

const fs = await import("node:fs");
fs.writeFileSync(out, Buffer.from(shot.data, "base64"));
console.log("已存", out, b.w + "x" + b.h, "@2x");
ws.close(); process.exit(0);
