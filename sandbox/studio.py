# -*- coding: utf-8 -*-
"""studio.py — 实验场控制台（可写）。与 view.py 的分工必须说清楚：

  view.py    只读观测：不写 runs/、不触发工具、不需要 key。评委可以一边看一边信。
  studio.py  控制台：能在页面上出题、选模型、真跑一轮 agent，所以它**必然**会写盘、会调工具。

两者的数据合同是同一个：runs/<tag>/ 下的 events.jsonl / steps.json / frames / result.json。
控制台不新增一种存储格式，跑出来的东西 view.py 照样能看、能导出、能进 reproduce.sh 的矩阵。

页面只有一张：viz/studio.html（零外部依赖），用 hash 路由分两页：
  #/        对赛事的理解 + 解法 + 40 个工具
  #/lab     三栏实验场：左=工具映射，中=对话与过程，右=画面/读写层/答案

控制台不认识任何一道题。出题入口只有两样东西：一段原话、一个模型文件。
「这段话要什么」由 agent/brief.py 当场编译（中间产物，随这一轮存档），
不是让人填的结构化表单 —— 填表会把「读懂题目」这一步替 agent 做掉。

用法：
  python studio.py                 # http://127.0.0.1:8770/
  python studio.py --port 9000
  python studio.py export --out ../docs/demo/studio.html   # 单文件快照（含数据与画面，无需服务）
"""
import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from view import load_run, summarize, run_dirs, RUNS, _jsonl, _load  # noqa: E402

HTML = os.path.join(HERE, "viz", "studio.html")
CARDS = os.path.join(RUNS, "_cards")
UPLOADS = os.path.join(RUNS, "_uploads")
MODEL = os.path.join(HERE, "..", "pkg", "AI-Geometry-Models", "IFC", "AI_Geometry_Input.ifc")

# 工具分组。分组只影响「人怎么找得到」，不影响工具能干什么 —— 一个工具可以出现在
# 多个阶段的白名单里（例如 profile 在识别/测量/复核都能用），这里按它回答的问题归类。
GROUPS = [
    ("读入与识别", "把文件变成实体，再把实体压到唯一命中", [
        "load", "clusters", "principal_frame", "select", "pick", "merge",
        "contains", "gaps", "parts", "levels", "segment"]),
    ("测量与判据", "每个数字都带见证物；判据要么有值要么明确说不知道", [
        "distance", "rank", "section", "profile", "mesh_health", "footprint",
        "clearance", "opening_edge", "pockets", "regid2d", "proj_area",
        "fit_sphere", "solve_section_height", "calc", "sphere_region", "contact_area"]),
    ("写层", "改几何的五个原语，改完必须落盘", [
        "translate", "stretch_axis", "cut_plane", "rotate_axis", "reopen_round", "emit"]),
    ("运动", "连续过程：撞没撞上、第一次接触在哪一度", [
        "collide", "sweep_rotate", "sweep_translate"]),
    ("提交", "答案与它的证据清单一起交", ["answer"]),
]

# 这里刻意没有「题目清单」。控制台不认识任何一道题：输入只有一段原话和一个模型文件，
# 结构化部分是 agent/brief.py 当场编译出来的中间产物。有任何一张内置题目表，
# 就等于提前把「读懂题目」这一步替 agent 做掉了。

_LIVE = {}          # tag -> {"proc": Popen, "t0": ts, "brief": path, "model": path, "llm": str}
_LOCK = threading.Lock()
TAG_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


# ------------------------------------------------------------------ 元数据

def tool_catalog():
    import tools.ops  # noqa: F401  注册所有工具
    from core.runtime import TOOLS
    from agent.loop import PHASE_TOOLS

    phases = {}
    for ph, names in PHASE_TOOLS.items():
        for n in names:
            phases.setdefault(n, []).append(ph)
    order = {n: i for i, n in enumerate(
        [n for _, _, ns in GROUPS for n in ns])}
    rows = []
    for name, meta in TOOLS.items():
        rows.append({
            "name": name,
            "desc": meta.get("desc") or "",
            "params": meta.get("params") or {},
            "pre": meta.get("pre") or "",
            "degrade": meta.get("degrade") or "",
            "returns": meta.get("returns") or "",
            "unit": meta.get("unit") or "",
            "phases": [p for p in ("SURVEY", "IDENTIFY", "MEASURE", "VERIFY",
                                   "EDIT_PLAN", "EDIT", "REVALIDATE", "REPORT") if p in phases.get(name, [])],
            "group": next((g for g, _, ns in GROUPS if name in ns), "其他"),
        })
    rows.sort(key=lambda r: (order.get(r["name"], 999), r["name"]))
    return rows


def home_payload():
    """主页要的东西。

    注意这里没有 tasks 字段：控制台不认识任何一道题，也不提供「挑一道题来试试」。
    主页给的是工具、能力分组、以及过去跑过的运行（供回看），仅此而已。
    """
    runs = [summarize(t, d) for t, d in run_dirs()]
    key = [r for r in runs if r["kind"] == "key"]
    n_write = 0
    for r in key:
        d = os.path.join(RUNS, r["tag"])
        if os.path.isdir(d):
            n_write += sum(1 for f in os.listdir(d) if f.endswith((".gltf", ".obj")))
    return {
        "tools": tool_catalog(),
        "groups": [{"name": g, "note": note, "tools": ns} for g, note, ns in GROUPS],
        "stats": {
            "n_tools": len(tool_catalog()),
            "n_checks": 81,
            "n_runs": len(key),
            "n_events": sum(r["n_events"] for r in key),
            "n_write_outputs": n_write,
        },
        "runs": runs,
    }


def models():
    out = []
    cand = [
        ("赛方原件 · IFC", "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc"),
        ("同一模型 · 转 OBJ 平面", "variants/obj_plain.obj"),
        ("同一模型 · 绕竖轴 90°", "variants/obj_rot90.obj"),
        ("同一模型 · 绕竖轴 45°", "variants/obj_rot45.obj"),
        ("同一模型 · 毫米单位", "variants/obj_mm.obj"),
    ]
    for label, rel in cand:
        p = os.path.normpath(os.path.join(HERE, rel))
        out.append({"label": label, "path": rel, "exists": os.path.exists(p),
                    "size": os.path.getsize(p) if os.path.exists(p) else None})
    for fn in sorted(os.listdir(UPLOADS)) if os.path.isdir(UPLOADS) else []:
        p = os.path.join(UPLOADS, fn)
        out.append({"label": "上传 · " + fn, "path": os.path.relpath(p, HERE).replace("\\", "/"),
                    "exists": True, "size": os.path.getsize(p)})
    return out


# ------------------------------------------------------------------ 出题 / 跑

def start_solve(text, model_rel, title=None, max_turns=70, wall=900,
                grace_turns=None, grace_wall=None):
    """一段原话 + 一个模型文件 -> 起一轮 agent。

    页面上能填的只有这两样。把「这段话要什么」读出来是这条管线自己的活：
    `agent.brief.compile_brief` 把原话编成本轮的任务规格（交付类型、要交的量、题面点名的实体、
    原话自带的约束），再由它渲染成注入 LLM 的说明。规格会落在 runs/_cards/<tag>.brief.json，
    跟着这一轮一起存档 —— 它是中间产物，不是输入格式。

    预算分两层：max_turns/wall 是**声明预算**（驱动收口提示），grace_turns/grace_wall 是
    **容错额度**（终止预算 = 声明 + 容错，真正硬停）。不传容错则 run_agent 按 1.5×/最低+10 自动算。
    """
    text = (text or "").strip()
    if len(text) < 8:
        raise ValueError("题目太短：至少给一句完整的要求")
    if len(text) > 20000:
        raise ValueError("题目太长：单轮上限 20000 字")

    model_abs = os.path.normpath(os.path.join(HERE, model_rel))
    if not os.path.exists(model_abs):
        raise ValueError("模型文件不存在：%s" % model_rel)

    from agent.brief import compile_brief
    # title 是给页面列表看的短标签，不是题目的一部分。不能拼进 raw ——
    # 否则 natural-language 整理会把「题：xxx」印成「题：<title> 题目」，答非所问。
    # compile_brief 只处理纯题目原文；title 单独存档（brief.json 的 title 字段）。
    brief = compile_brief(text)
    if title:
        brief["title"] = title.strip()

    stamp = time.strftime("%m%d_%H%M%S")
    tag = "live_%s" % stamp
    with _LOCK:
        n = 1
        while os.path.exists(os.path.join(RUNS, tag)):
            n += 1
            tag = "live_%s_%d" % (stamp, n)

    os.makedirs(CARDS, exist_ok=True)
    brief_path = os.path.join(CARDS, tag + ".brief.json")
    with open(brief_path, "w", encoding="utf-8") as f:
        json.dump(brief, f, ensure_ascii=False, indent=1)

    outdir = os.path.join(RUNS, tag)
    os.makedirs(outdir, exist_ok=True)

    llm = os.environ.get("STUDIO_LLM") or (
        "deepseek-chat" if os.environ.get("DEEPSEEK_API_KEY") else "gpt-4o")
    if llm.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "本机没有可用的 LLM key（DEEPSEEK_API_KEY / OPENAI_API_KEY 都没设）。"
            "这套东西不内置题目、也不回放别人的解题记录：没有 key 就只有工具层，没有决策层。")

    cmd = [sys.executable, "-u", "run_agent.py", "--brief", os.path.relpath(brief_path, HERE),
           "--model", model_rel, "--tag", tag, "--orbits", "1", "--no-video",
           "--max-turns", str(int(max_turns)), "--wall-budget", str(float(wall)),
           "--llm", llm]
    if grace_turns is not None:
        cmd += ["--grace-turns", str(int(grace_turns))]
    if grace_wall is not None:
        cmd += ["--grace-wall", str(float(grace_wall))]
    note = ("在线 LLM：%s（temperature=0）· 任务规格由原话编译：%d 个量 / %d 个实体"
            % (llm, len(brief["quantities"]), len(brief["subjects"])))

    log = open(os.path.join(outdir, "console.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _LIVE[tag] = {"proc": proc, "t0": time.time(), "brief": brief_path,
                  "model": model_rel, "mode": "llm", "note": note,
                  "llm": llm, "title": (title or "").strip()}
    return {"tag": tag, "note": note, "cmd": " ".join(cmd),
            "brief": {k: brief[k] for k in ("kind", "kind_label", "quantities",
                                            "subjects", "rules", "prompt")}}


def live_state(tag, since=0):
    if not TAG_RE.match(tag or ""):
        raise KeyError(tag)
    d = os.path.join(RUNS, tag)
    if not os.path.isdir(d):
        raise KeyError(tag)
    ev = _jsonl(os.path.join(d, "events.jsonl"))
    info = _LIVE.get(tag) or {}
    proc = info.get("proc")
    running = bool(proc and proc.poll() is None)
    res_p = os.path.join(d, "result.json")
    done = os.path.exists(res_p)
    status = "RUNNING"
    for e in reversed(ev):
        if e.get("kind") == "agent_end":
            status = (e.get("payload") or {}).get("status") or status
            break
    if done:
        try:
            status = (_load(res_p, {}) or {}).get("status") or status
        except Exception:
            pass
    elif not running and proc is not None:
        status = "EXITED"
    return {
        "tag": tag, "running": running, "done": done, "status": status,
        "since": since, "n": len(ev),
        "events": ev[since:],
        "elapsed_s": round(time.time() - info["t0"], 1) if info else None,
        "note": info.get("note"), "title": info.get("title"),
        "llm": info.get("llm"), "frames": len(_load(os.path.join(d, "frames", "manifest.json"), []) or []),
        # 进行中的单帧预览（run_agent.live_preview 写的）：页面用它让「几何画面」
        # 在跑的过程中就有东西可看，而不是一块白板 + 一个破图图标。
        "preview": _load(os.path.join(d, "frames", "live.json"), None),
        "brief": _load(os.path.join(d, "brief.json"), None),
    }


# ------------------------------------------------------------------ 存档

def _clean_paths(o):
    """<repo>\\sandbox\\..\\pkg\\x 这种由相对路径拼出来的 .. 对看的人没有意义。
    在解析后的字符串上做替换，避免在 JSON 文本里数反斜杠（数错过一次）。"""
    if isinstance(o, str):
        # args 是先 json.dumps 又嵌进 result.json 的，字符串里可能是 1~3 层反斜杠，
        # 逐层都换一遍；只换最外层会漏掉深层那一份。
        out = o
        for _ in range(3):
            out = out.replace("\\\\sandbox\\\\..\\\\", "\\\\").replace("\\sandbox\\..\\", "\\")
        return out.replace("/sandbox/../", "/")
    if isinstance(o, dict):
        return {k: _clean_paths(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean_paths(v) for v in o]
    return o


def run_payload(tag):
    """在 view.load_run 的基础上补两样东西：原话，和这一轮编译出来的任务规格。

    观测台只放卡片名就够了（它默认你知道题），但控制台要让人看见「agent 到底收到了什么」，
    以及「那句话是被怎么读成任务的」。规格本来就在 runs/<tag>/brief.json 里，
    这里只是把它带出去给页面渲染。
    """
    r = load_run(tag)
    # 页面会被投屏、会被截图、会被贴进群里：本机绝对路径不该跟着出去。
    # 只对「发出去的这一份」做替换，磁盘上的账本原样保留（审计要能对回原文）。
    from view import _sanitize_paths
    r = json.loads(_sanitize_paths(json.dumps(r, ensure_ascii=False, default=str)))
    r = _clean_paths(r)
    env = (r.get("result") or {}).get("env") or {}

    bp = env.get("brief")
    if not bp:
        cand = os.path.join(RUNS, tag, "brief.json")
        bp = cand if os.path.exists(cand) else None
    if bp:
        p = os.path.normpath(os.path.join(HERE, bp))
        if os.path.exists(p) and os.path.commonpath([p, HERE]) == HERE:
            try:
                brief = json.load(open(p, encoding="utf-8"))
                r["meta"]["card"] = "本轮任务（原话 + 管线编译）"
                r["meta"]["card_text"] = brief.get("raw") or ""
                r["meta"]["brief"] = {k: brief.get(k) for k in
                                      ("raw", "kind", "kind_label", "quantities",
                                       "subjects", "rules", "prompt", "chars")}
            except Exception:
                pass
        return r

    # 老运行（任务卡路径）：把卡片正文带上，注明它是题卡而不是本轮原话。
    rel = env.get("card") or ""
    if rel:
        p = os.path.normpath(os.path.join(HERE, rel))
        if os.path.exists(p) and os.path.commonpath([p, HERE]) == HERE:
            try:
                r["meta"]["card_text"] = open(p, encoding="utf-8").read()[:6000]
            except Exception:
                pass
    return r


def archive(tag, inline=True):
    r = run_payload(tag)
    out = {"kind": "hack26.studio.run", "v": 1, "tag": tag,
           "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "run": r}
    if inline:
        import base64
        frames = {}
        for f in r["frames"]:
            fp = os.path.join(RUNS, tag, "frames", "seq_%03d.png" % f["i"])
            if os.path.exists(fp):
                with open(fp, "rb") as fh:
                    frames[str(f["i"])] = "data:image/png;base64," + base64.b64encode(fh.read()).decode()
        out["frames"] = frames
    return out


# ------------------------------------------------------------------ HTTP

CTYPE = {".png": "image/png", ".mp4": "video/mp4",
         ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8",
         ".md": "text/plain; charset=utf-8", ".gltf": "model/gltf+json",
         ".js": "application/javascript; charset=utf-8"}


class H(BaseHTTPRequestHandler):
    server_version = "studio"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 4 << 20:
            raise ValueError("请求体为空或过大")
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                with open(HTML, encoding="utf-8") as f:
                    return self._send(200, f.read(), CTYPE[".html"])
            if u.path == "/api/home":
                return self._send(200, home_payload())
            if u.path == "/api/models":
                return self._send(200, {"models": models()})
            if u.path == "/api/runs":
                return self._send(200, {"runs": [summarize(t, d) for t, d in run_dirs()]})
            if u.path == "/api/run":
                return self._send(200, run_payload(os.path.basename((q.get("tag") or [""])[0])))
            if u.path == "/api/live":
                tag = os.path.basename((q.get("tag") or [""])[0])
                since = int((q.get("since") or ["0"])[0])
                return self._send(200, live_state(tag, since))
            if u.path == "/api/geom":
                tag = os.path.basename((q.get("tag") or [""])[0])
                if not TAG_RE.match(tag or ""):
                    raise KeyError(tag)
                fp = os.path.join(RUNS, tag, "geom.json")
                if not os.path.exists(fp):
                    return self._send(404, {"error": "no geometry",
                                            "hint": "这一轮跑的时候还没有几何导出，选一轮新的"})
                if (q.get("probe") or [""])[0] == "1":
                    # 运行中几何每几秒就变一次；页面 800 ms 轮询一次，
                    # 每次都拉 500 KB 是纯浪费。先问 mtime，变了才取全量。
                    st = os.stat(fp)
                    return self._send(200, {"mtime": round(st.st_mtime, 3), "bytes": st.st_size})
                with open(fp, "rb") as f:
                    body = f.read()
                # 几何是扁平数字数组，gzip 后只有 ~1/4；本机也省一次拷贝
                if "gzip" in (self.headers.get("Accept-Encoding") or ""):
                    body = gzip.compress(body, 6)
                    return self._send(200, body, CTYPE[".json"], extra={"Content-Encoding": "gzip"})
                return self._send(200, body, CTYPE[".json"])
            if u.path == "/api/archive":
                tag = os.path.basename((q.get("tag") or [""])[0])
                inline = (q.get("inline") or ["1"])[0] != "0"
                return self._send(200, archive(tag, inline))
            if u.path.startswith("/video/"):
                tag = os.path.basename(u.path[len("/video/"):])
                full = os.path.join(RUNS, tag, "video.mp4")
                if not os.path.exists(full):
                    return self._send(404, {"error": "no video"})
                with open(full, "rb") as f:
                    return self._send(200, f.read(), CTYPE[".mp4"])
            if u.path.startswith("/lib/"):
                rel = os.path.basename(u.path[len("/lib/"):])
                full = os.path.join(HERE, "viz", "lib", rel)
                if not os.path.exists(full):
                    return self._send(404, {"error": "not found"})
                with open(full, "rb") as f:
                    return self._send(200, f.read(), CTYPE.get(".js"))
            if u.path.startswith("/frames/"):
                rel = u.path[len("/frames/"):]
                full = os.path.normpath(os.path.join(RUNS, rel))
                if not full.startswith(RUNS):
                    return self._send(403, {"error": "forbidden"})
                if not os.path.exists(full):
                    return self._send(404, {"error": "not found"})
                ext = os.path.splitext(full)[1].lower()
                with open(full, "rb") as f:
                    return self._send(200, f.read(), CTYPE.get(ext, "application/octet-stream"))
            return self._send(404, {"error": "not found"})
        except KeyError:
            return self._send(404, {"error": "no such run"})
        except Exception as ex:
            return self._send(500, {"error": "%s: %s" % (type(ex).__name__, ex)})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/solve":
                b = self._json_body()
                gt = b.get("grace_turns")
                gw = b.get("grace_wall")
                r = start_solve(b.get("text"),
                                b.get("model") or "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc",
                                b.get("title"),
                                int(b.get("max_turns") or 70), float(b.get("wall") or 900),
                                int(gt) if gt not in (None, "") else None,
                                float(gw) if gw not in (None, "") else None)
                return self._send(200, r)
            if u.path == "/api/upload":
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > 80 << 20:
                    return self._send(400, {"error": "文件为空或超过 80 MB"})
                name = os.path.basename(self.headers.get("X-Filename") or "upload.bin")
                name = re.sub(r"[^\w.\-]", "_", name)[:80]
                os.makedirs(UPLOADS, exist_ok=True)
                stamp = time.strftime("%m%d_%H%M%S")
                dst = os.path.join(UPLOADS, "%s_%s" % (stamp, name))
                with open(dst, "wb") as f:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                return self._send(200, {"path": "runs/_uploads/" + os.path.basename(dst),
                                        "size": os.path.getsize(dst)})
            return self._send(404, {"error": "not found"})
        except ValueError as ex:
            return self._send(400, {"error": str(ex)})
        except Exception as ex:
            return self._send(500, {"error": "%s: %s" % (type(ex).__name__, ex)})


# ------------------------------------------------------------------ 导出

def cmd_export(a):
    """把主页 + 若干 run（含帧）塞进一个 HTML：双击即看，出题入口降级为只读。"""
    tags = [t.strip() for t in (a.runs or "").split(",") if t.strip()]
    tags = [t for t in tags if os.path.isdir(os.path.join(RUNS, t))]
    payload = {"home": home_payload(), "runs": {}, "frames": {}, "live": False}
    for t in tags:
        payload["runs"][t] = run_payload(t)
    payload["geom"] = {}
    for t in tags:
        gp = os.path.join(RUNS, t, "geom.json")
        if os.path.exists(gp):
            payload["geom"][t] = _load(gp, None)
    if a.inline:
        import base64
        for t in tags:
            frames = {}
            for f in payload["runs"][t]["frames"]:
                fp = os.path.join(RUNS, t, "frames", "seq_%03d.png" % f["i"])
                if os.path.exists(fp):
                    with open(fp, "rb") as fh:
                        frames[str(f["i"])] = "data:image/png;base64," + base64.b64encode(fh.read()).decode()
            payload["frames"][t] = frames
    html = open(HTML, encoding="utf-8").read()
    # three.js 内联进单文件：导出的意义是「双击即看」，不能依赖本机还有一个 web 服务
    lib = os.path.join(HERE, "viz", "lib", "three.min.js")
    lib_js = ""
    if a.lib and os.path.exists(lib):
        with open(lib, encoding="utf-8") as fh:
            lib_js = "<script>%s</script>\n" % fh.read().replace("</script", "<\\/script")
    html = html.replace('<script src="/lib/three.min.js"></script>', lib_js)
    inject = '<script>window.__STUDIO__ = %s;</script>\n' % json.dumps(
        payload, ensure_ascii=False, default=str).replace("</", "<\\/")
    from view import _sanitize_paths
    inject = _sanitize_paths(inject)
    if "<script>\nlet BOOT" not in html:
        raise SystemExit("studio.html 缺少注入点（let BOOT）")
    html = html.replace("<script>\nlet BOOT", inject + "<script>\nlet BOOT", 1)
    out = a.out or os.path.join(HERE, "..", "docs", "demo", "studio.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("已导出 %s（%.1f MB，%d 个 run，%d 帧，%d 份几何）" % (
        out, os.path.getsize(out) / 1048576.0, len(tags),
        sum(len(v) for v in payload["frames"].values()), len(payload["geom"])))


def cmd_serve(a):
    if not os.path.exists(HTML):
        raise SystemExit("缺少 %s" % HTML)
    srv = ThreadingHTTPServer((a.host, a.port), H)
    url = "http://%s:%d/" % (a.host, a.port)
    print("实验场控制台 %s" % url)
    print("  主页 #/   实验场 #/lab")
    print("  与 view.py 的区别：这一层会真跑 agent（写 runs/<tag>/），view.py 保持只读。")
    print("  模型：%s" % ("存在" if os.path.exists(MODEL) else "缺失（出题前请先下载赛方模型包）"))
    print("  LLM：%s" % ("deepseek-chat（检测到 DEEPSEEK_API_KEY）"
                        if os.environ.get("DEEPSEEK_API_KEY") else "未检测到 key，出题请用离线模式"))
    if a.open:
        try:
            webbrowser.open(url + "#/")
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


def main():
    ap = argparse.ArgumentParser(description="实验场控制台（可出题、可回放）")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="本地控制台（默认）")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--open", action="store_true", default=True)
    s.add_argument("--no-open", dest="open", action="store_false")
    e = sub.add_parser("export", help="导出单文件快照")
    e.add_argument("--out")
    e.add_argument("--runs", default="ag_ifc,edit09,edit02",
                   help="逗号分隔的 run tag，默认带三题（写层一题、读层一题、T-01）")
    e.add_argument("--inline", action="store_true", default=True)
    e.add_argument("--no-inline", dest="inline", action="store_false")
    e.add_argument("--no-lib", dest="lib", action="store_false",
                   help="不内联 three.js（文件小 600 KB，但三维视图不可用）")
    a = ap.parse_args()
    if a.cmd == "export":
        cmd_export(a)
    else:
        if a.cmd is None:
            a = ap.parse_args(["serve"])
        cmd_serve(a)


if __name__ == "__main__":
    main()
