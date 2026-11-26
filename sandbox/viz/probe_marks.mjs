// 验证 marks 全链路：加载已完成 run → 确认 marks 数据 → 操作 UI → 检查叠加
const PORT = +(process.env.CDP_PORT || 9222);
const TAG = process.env.TAG || "live_1126_163956";
const raw = `http://127.0.0.1:8770/#/lab?tag=${TAG}`;
const [base0, hash0] = raw.split("#");
const URL_ = base0 + (base0.includes("?") ? "&" : "?") + "cb=" + Date.now() + (hash0 ? "#" + hash0 : "");
const tabs = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const page = tabs.find(t => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
const send = (m, p = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method: m, params: p })); });
const logs = [];
ws.onmessage = ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result || m.error); pending.delete(m.id); }
  if (m.method === "Runtime.consoleAPICalled") logs.push("[console." + m.params.type + "] " + m.params.args.map(a => a.value ?? a.description ?? a.type).join(" "));
  if (m.method === "Runtime.exceptionThrown") logs.push("[exception] " + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
};
await new Promise(r => ws.onopen = r);
await send("Runtime.enable"); await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1680, height: 1050, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: URL_ });

// 等页面完全加载（和 probe_diag.mjs 一致，8 秒）
await new Promise(r => setTimeout(r, 8000));

// 步骤 1：确认 marks 已加载
const initial = await send("Runtime.evaluate", { expression: `JSON.stringify({
  has_geo: !!(S.geo && S.geo.data),
  marks_n: S.geo && S.geo.data ? (S.geo.data.marks||[]).length : null,
  marks_keys: S.geo && S.geo.data ? Object.keys(S.geo.data) : [],
  markStep: VIEW.markStep,
  markbar_html: (document.getElementById("markbar")||{}).innerHTML?.slice(0,200)||"(empty)",
})`, returnByValue: true, awaitPromise: true });
console.log("STEP1 初始:", initial.result?.value);

const init = JSON.parse(initial.result?.value || "{}");
if (!init.marks_n) { console.log("FAIL: marks 未加载"); ws.close(); process.exit(1); }

// 步骤 2：检查 marksGroup 是否建立
const s2 = await send("Runtime.evaluate", { expression: `JSON.stringify({
  has_marksGroup: !!VIEW.marksGroup,
  n_children: VIEW.marksGroup? VIEW.marksGroup.children.length : 0,
  markStep: VIEW.markStep,
  focus_sids: VIEW.focusSids? [...VIEW.focusSids].slice(0,5) : null,
  slider_max: document.getElementById("mkslider")? +document.getElementById("mkslider").max : null,
})`, returnByValue: true, awaitPromise: true });
console.log("STEP2 叠加状态:", s2.result?.value);

// 步骤 3：切到第 0 步，看 marksGroup 变化
const s3 = await send("Runtime.evaluate", { expression: `
  setMarkStep(0);
  JSON.stringify({
    markStep: VIEW.markStep,
    has_marksGroup: !!VIEW.marksGroup,
    n_children: VIEW.marksGroup? VIEW.marksGroup.children.length : 0,
    focus_sids: VIEW.focusSids? [...VIEW.focusSids].slice(0,5) : null,
    slider_val: document.getElementById("mkslider")? +document.getElementById("mkslider").value : null,
  })`, returnByValue: true, awaitPromise: true });
console.log("STEP3 切第0步:", s3.result?.value);

// 步骤 4：切到中间步骤
const mid = Math.floor(init.marks_n / 2);
const s4 = await send("Runtime.evaluate", { expression: `
  setMarkStep(${mid});
  JSON.stringify({
    markStep: VIEW.markStep,
    n_children: VIEW.marksGroup? VIEW.marksGroup.children.length : 0,
    focus_sids: VIEW.focusSids? [...VIEW.focusSids].slice(0,5) : null,
    camera: VIEW.camera ? VIEW.camera.position.toArray().map(x=>+x.toFixed(2)) : null,
    target: VIEW.sph.target.toArray().map(x=>+x.toFixed(2)),
  })`, returnByValue: true, awaitPromise: true });
console.log("STEP4 切第" + mid + "步:", s4.result?.value);

// 步骤 5：切到最后一步，看焦点实体变化
const last = init.marks_n - 1;
const s5 = await send("Runtime.evaluate", { expression: `
  setMarkStep(${last});
  JSON.stringify({
    markStep: VIEW.markStep,
    n_children: VIEW.marksGroup? VIEW.marksGroup.children.length : 0,
    focus_sids: VIEW.focusSids? [...VIEW.focusSids].slice(0,5) : null,
    camera: VIEW.camera ? VIEW.camera.position.toArray().map(x=>+x.toFixed(2)) : null,
    target: VIEW.sph.target.toArray().map(x=>+x.toFixed(2)),
  })`, returnByValue: true, awaitPromise: true });
console.log("STEP5 切最后步:", s5.result?.value);

// 判定
const r2 = JSON.parse(s2.result?.value || "{}");
const r3 = JSON.parse(s3.result?.value || "{}");
const r4 = JSON.parse(s4.result?.value || "{}");
const pass = r2.has_marksGroup && r2.n_children > 0 && r3.markStep === 0 && r4.markStep === mid;
console.log("---");
console.log("RESULT:", pass ? "PASS marks UI + overlay 全链路工作" : "FAIL");
console.log("LOGS:", logs.length ? logs.join("\n") : "(none)");
ws.close();
process.exit(pass ? 0 : 1);
