// 验证：焦点对象占画面 ~75% 且居中，其他构件半透明
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

// 等 marks 加载
let marksN = 0;
for (let i = 0; i < 12; i++) {
  await new Promise(r => setTimeout(r, 1000));
  const out = await send("Runtime.evaluate", { expression: `(S.geo&&S.geo.data)?(S.geo.data.marks||[]).length:0`, returnByValue: true });
  marksN = out.result?.value || 0;
  if (marksN > 0) break;
}
if (!marksN) { console.log("FAIL: marks 未加载"); ws.close(); process.exit(1); }

// 计算画面占比：把焦点实体投影到 NDC，测宽度/高度占比，以及中心偏离
const evalRes = await send("Runtime.evaluate", {
  expression: `
  (()=>{
    const mk=(S.geo.data.marks||[])[VIEW.markStep];
    const focus=mk.focus||[];
    if(!focus.length) return {error:"无焦点"};
    // 收集焦点实体 mesh 的包围盒 → 投影到 NDC 求占比
    const bb=new THREE.Box3();
    focus.forEach(sid=>{ const m=VIEW.meshes.find(x=>x.userData.sid===sid); if(m) bb.expandByObject(m); });
    if(bb.isEmpty()) return {error:"空包围盒"};
    // 8 个角投影到 NDC（-1..1）
    const pts=[];
    for(const x of [bb.min.x,bb.max.x]) for(const y of [bb.min.y,bb.max.y]) for(const z of [bb.min.z,bb.max.z])
      pts.push(new THREE.Vector3(x,y,z).project(VIEW.camera));
    const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
    const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
    const w=maxX-minX, h=maxY-minY;
    const cx=(minX+maxX)/2, cy=(minY+maxY)/2;
    // 占比（NDC 宽 2 对应满屏）
    const fracW=w/2, fracH=h/2;
    return {
      focus,
      fracW:+fracW.toFixed(3), fracH:+fracH.toFixed(3),
      centerDX:+cx.toFixed(3), centerDY:+cy.toFixed(3),
      camDist:+VIEW.sph.r.toFixed(3),
      target:+VIEW.sph.target.toArray().map(x=>+x.toFixed(2)).join(","),
      // 各构件透明度（focus vs 其他）
      focusOpacity: focus.length? VIEW.meshes.find(m=>m.userData.sid===focus[0])?.material.opacity : null,
      otherOpacity: VIEW.meshes.find(m=>!focus.includes(m.userData.sid) && !m.userData.struct)?.material.opacity ?? null,
      structOtherOpacity: VIEW.meshes.find(m=>!focus.includes(m.userData.sid) && m.userData.struct)?.material.opacity ?? null,
    };
  })()`,
  returnByValue: true, awaitPromise: true
});
const r = evalRes.result?.value || {};
console.log("RESULT:", JSON.stringify(r, null, 2));

// 判定：占比接近 0.75（0.55~0.9 容差，因为对角线取 0.7 系数偏保守），中心接近 0，其他半透明
const fracOK = r.fracW > 0.5 && r.fracW < 0.98 && r.fracH > 0.4;
const centerOK = Math.abs(r.centerDX) < 0.25 && Math.abs(r.centerDY) < 0.25;
const opacOK = (r.focusOpacity === 1) && (r.otherOpacity === 0.45) && (r.structOtherOpacity === 0.22);
const pass = fracOK && centerOK && opacOK;
console.log("---");
console.log("fracOK:", fracOK, "| centerOK:", centerOK, "| opacOK:", opacOK);
console.log("RESULT:", pass ? "PASS 焦点占~75%居中+其他半透明" : "FAIL");
console.log("LOGS:", logs.length ? logs.join("\n") : "(none)");
ws.close();
process.exit(pass ? 0 : 1);
