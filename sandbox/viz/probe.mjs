// 用 CDP 打开页面并读回状态：为什么 deep link / 默认运行没载入？
const PORT = 9222;
const raw = process.argv[2] || "http://127.0.0.1:8770/#/lab?tag=live_fix1";
// 同一个 URL 再 navigate 不会重新加载（hash 变了才触发 hashchange），加一次性参数强制刷新
const [base0, hash0] = raw.split("#");
const URL_ = base0 + (base0.includes("?") ? "&" : "?") + "cb=" + Date.now() + (hash0 ? "#" + hash0 : "");
const EXPR = process.argv[3] || "JSON.stringify({run: !!S.run, tag: (S.run||{}).tag, status: document.getElementById('hstat').innerText.trim(), shot: (document.getElementById('shotwrap')||{}).innerText, runs: (S.runs||[]).length})";

const tabs = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const page = tabs.find(t => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const send = (method, params = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const logs = [];

ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result || m.error); pending.delete(m.id); }
  if (m.method === "Runtime.consoleAPICalled")
    logs.push("[console." + m.params.type + "] " + m.params.args.map(a => a.value ?? a.description ?? a.type).join(" "));
  if (m.method === "Runtime.exceptionThrown")
    logs.push("[exception] " + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
};

await new Promise(r => ws.onopen = r);
await send("Runtime.enable");
await send("Page.enable");
await send("Page.navigate", { url: URL_ });
await new Promise(r => setTimeout(r, 7000));
const out = await send("Runtime.evaluate", { expression: EXPR, returnByValue: true, awaitPromise: true });
console.log("RESULT:", out.result?.value ?? JSON.stringify(out));
console.log("LOGS:", logs.length ? logs.join("\n") : "(none)");
ws.close();
process.exit(0);
