# -*- coding: utf-8 -*-
"""view.py — 实验场的人可观测层（零依赖）。

为什么需要它：沙盒跑完只留下 events.jsonl / steps.json / frames，评委得自己拼。
「可观测」不是把日志 cat 出来，而是能回答三个问题：
  1) agent 在第几步、什么阶段、为什么调这个工具；
  2) 这个数字是哪一步、哪个见证点给出的（追得到证据行）；
  3) 账本有没有被改过（哈希链是否连续）。

两种形态：
  python view.py                 # 本地观测台：起一个 127.0.0.1 的只读服务并开浏览器
  python view.py ls              # 终端列表：所有 run 的状态/轮次/答案
  python view.py show ag_ifc     # 终端时间线：逐步打印决策链（可 --full 看原始观察）

只读：不写 runs/、不触发任何工具、不需要 API key。看的人和跑的人可以是两个人。
"""
import argparse
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
HTML = os.path.join(HERE, "viz", "viewer.html")

PHASES = ["SURVEY", "IDENTIFY", "MEASURE", "VERIFY", "EDIT", "REVALIDATE", "REPORT"]
C = {"SURVEY": "\033[36m", "IDENTIFY": "\033[35m", "MEASURE": "\033[33m",
     "VERIFY": "\033[32m", "EDIT": "\033[38;5;166m", "REVALIDATE": "\033[38;5;37m",
     "RECOVER": "\033[31m", "REPORT": "\033[37m"}
OKC = {"OK": "\033[42;30m", "GATE_BLOCKED": "\033[43;30m", "AUDIT_FAILED": "\033[41;37m",
       "STALLED": "\033[45;37m", "BUDGET_EXHAUSTED": "\033[45;37m",
       "INSUFFICIENT_EVIDENCE": "\033[46;30m"}
R = "\033[0m"
DIM = "\033[2m"


# ---------------------------------------------------------------- 读取

def run_dirs():
    if not os.path.isdir(RUNS):
        return []
    out = []
    for name in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "result.json")):
            out.append((name, d))
    return out


def _load(path, dflt=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return dflt


def _jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        pass
    return rows


def _norm_answer(res):
    """把两条执行路径的答案形状归一：agent 给 distances_m（dict），plan 执行器给
    distance_ranking（list）。展示层不该让看的人去分辨这个差别。"""
    a = res.get("answer")
    if not isinstance(a, dict):
        return res
    v = dict(a.get("values") or {})
    if not v.get("distances_m") and isinstance(v.get("distance_ranking"), list):
        v["distances_m"] = {r.get("sid"): r.get("value") for r in v["distance_ranking"]
                            if isinstance(r, dict) and r.get("sid")}
        a = dict(a, values=v)
        res = dict(res, answer=a)
    return res


def summarize(tag, d):
    res = _norm_answer(_load(os.path.join(d, "result.json"), {}) or {})
    ev = _jsonl(os.path.join(d, "events.jsonl"))
    man = _load(os.path.join(d, "frames", "manifest.json"), []) or []
    start = next((e["payload"] for e in ev if e.get("kind") == "agent_start"), {})
    ans = res.get("answer") or {}
    vals = ans.get("values") or {}
    led = None
    for e in reversed(ev):
        if e.get("kind") == "ledger_check" and isinstance(e.get("payload"), dict):
            led = e["payload"]
            break
    return {
        "tag": tag,
        "status": res.get("status") or "DONE",
        "turns": res.get("turns"),
        "elapsed_s": res.get("elapsed_s"),
        "phase": res.get("phase"),
        "card": os.path.basename(start.get("card") or ""),
        "model": os.path.basename(start.get("model") or ""),
        "policy": (start.get("policy") or ""),
        "n_events": len(ev),
        "n_frames": len(man),
        "has_video": os.path.exists(os.path.join(d, "video.mp4")),
        "has_report": os.path.exists(os.path.join(d, "report.md")),
        "ledger_ok": (led or {}).get("ok"),
        "ledger_n": (led or {}).get("n"),
        "answer": vals or None,
        "mtime": os.path.getmtime(os.path.join(d, "result.json")),
        "kind": "dev" if (tag.startswith("_") or tag.startswith("llm")) else "key",
    }


def load_run(tag):
    d = os.path.join(RUNS, tag)
    if not os.path.isdir(d):
        raise KeyError(tag)
    res = _norm_answer(_load(os.path.join(d, "result.json"), {}) or {})
    steps = _load(os.path.join(d, "steps.json"), []) or []
    events = _jsonl(os.path.join(d, "events.jsonl"))
    man = _load(os.path.join(d, "frames", "manifest.json"), []) or []
    report = ""
    p = os.path.join(d, "report.md")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            report = f.read()
    # 账本链式自检（只验结构：本条 prev 是否等于上一条 hash；不重算 sha，避免把展示层做成信任根）
    prev, breaks = None, []
    for i, e in enumerate(events):
        if i and e.get("prev") != prev:
            breaks.append(i)
        prev = e.get("hash")
    return {
        "tag": tag,
        "meta": summarize(tag, d),
        "result": {k: v for k, v in res.items() if k != "steps"},
        "steps": steps,
        "events": events,
        "frames": man,
        "report": report,
        "chain_breaks": breaks,
        "phases": PHASES,
    }


# ---------------------------------------------------------------- 终端形态

def _bar(status):
    return "%s %s %s" % (OKC.get(status, "\033[100m"), " %-20s " % status, R)


def cmd_ls(a):
    runs = [summarize(t, d) for t, d in run_dirs()]
    if a.kind != "all":
        runs = [r for r in runs if r["kind"] == a.kind]
    runs.sort(key=lambda r: (r["kind"] != "key", -(r["mtime"] or 0)))
    print("%-22s %-22s %5s %7s %6s %5s  %s" % ("tag", "状态", "轮次", "耗时s", "事件", "帧", "答案/备注"))
    print("-" * 112)
    for r in runs:
        ans = ""
        v = (r.get("answer") or {})
        d = v.get("distances_m") or {}
        if d:
            ans = "chair=%s  " % v.get("chair") + "  ".join("%s=%.3f" % (k, x) for k, x in d.items())
        elif r["status"] != "OK":
            ans = DIM + "（未提交答案）" + R
        led = "" if r["ledger_ok"] in (True, None) else "  账本:断"
        print("%-22s %-22s %5s %7s %6d %5d  %s%s" % (
            r["tag"], _bar(r["status"]), r["turns"], "%.2f" % (r["elapsed_s"] or 0),
            r["n_events"], r["n_frames"], ans, led))
    print("\n共 %d 个 run。看时间线：python view.py show <tag>；看画面：python view.py" % len(runs))


def cmd_show(a):
    r = load_run(a.tag)
    m = r["meta"]
    print("\n%s  %s  %s" % (m["tag"], _bar(m["status"]), DIM + (m["card"] or "") + R))
    print("%s轮次 %s · 耗时 %ss · 事件 %d · 账本 %s%s" % (
        DIM, m["turns"], m["elapsed_s"], m["n_events"],
        "完好" if not r["chain_breaks"] else "第 %s 条断裂" % r["chain_breaks"], R))
    print("-" * 100)
    steps = {s.get("id"): s for s in r["steps"]}
    for e in r["events"]:
        k, p = e.get("kind"), e.get("payload") or {}
        if k == "decision":
            ph = p.get("phase") or ""
            print("%s%s%s %-9s %-14s %s" % (C.get(ph, ""), ph[:9].ljust(9), R,
                                            p.get("step"), p.get("tool"),
                                            DIM + (p.get("why") or "")[:150] + R))
        elif k == "tool_call":
            st = steps.get(p.get("step")) or {}
            n = st.get("n")
            print("   %s%s %5.1fms  %-18s %s%s" % (
                "\033[32m✓\033[0m" if p.get("ok") else "\033[31m✗\033[0m",
                "", p.get("ms") or 0, "hit=%s" % n if n is not None else "",
                DIM + (p.get("result_excerpt") or "")[:120] + R, ""))
            if a.full and st.get("obs"):
                print(DIM + "     obs: " + st["obs"][:a.full] + R)
        elif k == "phase":
            print("%s→ %s：%s%s" % (DIM, p.get("to"), (p.get("why") or "")[:110], R))
        elif k in ("gate_reject", "resolve_error", "audit_reject", "stall_abort",
                   "failure_raised", "failure_resolved", "role_decided", "label_renamed"):
            print("\033[33m· %s: %s\033[0m" % (k, json.dumps(p, ensure_ascii=False)[:160]))
        elif k == "agent_end":
            print("%s· end: %s\033[0m" % (OKC.get(p.get("status"), ""), json.dumps(p, ensure_ascii=False)[:120]))
    d = ((r["result"].get("answer") or {}).get("values") or {})
    if d:
        print("-" * 100)
        print("答案：chair=%s  %s" % (d.get("chair"),
                                 "  ".join("%s=%.6f m" % (k, v) for k, v in (d.get("distances_m") or {}).items())))
    if r["report"]:
        print(DIM + "报告：runs/%s/report.md · 账本：runs/%s/events.jsonl · 画面：python view.py" % (a.tag, a.tag) + R)


# ---------------------------------------------------------------- 服务形态

CTYPE = {".png": "image/png", ".mp4": "video/mp4", ".html": "text/html; charset=utf-8",
         ".json": "application/json; charset=utf-8", ".md": "text/plain; charset=utf-8"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                with open(HTML, encoding="utf-8") as f:
                    return self._send(200, f.read(), CTYPE[".html"])
            if u.path == "/api/runs":
                return self._send(200, {"runs": [summarize(t, d) for t, d in run_dirs()]})
            if u.path == "/api/run":
                tag = (q.get("tag") or [""])[0]
                tag = os.path.basename(tag)
                return self._send(200, load_run(tag))
            if u.path.startswith("/video/"):
                tag = os.path.basename(u.path[len("/video/"):])
                full = os.path.join(RUNS, tag, "video.mp4")
                if not os.path.exists(full):
                    return self._send(404, {"error": "no video"})
                with open(full, "rb") as f:
                    return self._send(200, f.read(), CTYPE[".mp4"])
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
        except BrokenPipeError:
            pass
        except Exception as ex:  # 观测层永远不该把主流程搞挂
            return self._send(500, {"error": "%s: %s" % (type(ex).__name__, ex)})


def _sanitize_paths(text):
    """把绝对路径里的用户名/目录抹掉。

    坑：观察本身是「JSON 里的字符串」，塞进 payload 后路径的反斜杠会被再转义一层
    （C:\\Users → C:\\\\Users）。所以按 1~4 层转义各换一遍，别只换最外层。
    """
    home = os.path.expanduser("~")
    root = os.path.dirname(HERE)          # 仓库根
    for depth in range(1, 5):
        for real, alias in ((root, "<repo>"), (home, "~")):
            form = real
            for _ in range(depth - 1):
                form = form.replace("\\", "\\\\")
            if len(form) > 3:
                text = text.replace(form, alias)
                text = text.replace(form.replace("\\", "/"), alias)
    return text


def cmd_export(a):
    """把一个 run 连同数据、画面导成单文件 HTML：双击即看，不需要起服务、不需要环境。"""
    r = load_run(a.tag)
    with open(HTML, encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps(r, ensure_ascii=False).replace("</", "<\\/")
    inject = '<script>window.__RUN__ = %s;</script>\n' % payload
    if a.inline:
        import base64
        frames = {}
        for f in r["frames"]:
            fp = os.path.join(RUNS, a.tag, "frames", "seq_%03d.png" % f["i"])
            if os.path.exists(fp):
                with open(fp, "rb") as fh:
                    frames[str(f["i"])] = "data:image/png;base64," + base64.b64encode(fh.read()).decode()
        inject += '<script>window.__FRAMES__ = %s;</script>\n' % json.dumps(frames)
    # 导出的页面常被贴到仓库/分享给人看：把本机绝对路径换成 ~ / <repo>，别带上自己的用户名
    inject = _sanitize_paths(inject)
    html = html.replace("<script>\nconst EMB", inject + "<script>\nconst EMB", 1)
    if "window.__RUN__" not in html:
        raise SystemExit("viewer.html 缺少注入点（const EMB 行）")
    out = a.out or os.path.join(RUNS, a.tag, "viewer.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    kb = os.path.getsize(out) / 1024.0
    print("已导出 %s（%.0f KB，内含 %d 事件 / %d 帧）" % (out, kb, len(r["events"]), len(r["frames"])))
    if a.open:
        try:
            webbrowser.open("file:///" + os.path.abspath(out).replace("\\", "/"))
        except Exception:
            pass


def cmd_serve(a):
    if not os.path.exists(HTML):
        raise SystemExit("缺少 %s" % HTML)
    srv = ThreadingHTTPServer((a.host, a.port), H)
    url = "http://%s:%d/" % ("127.0.0.1" if a.host == "0.0.0.0" else a.host, a.port)
    print("观测台 %s  （Ctrl-C 退出，只读）" % url)
    print("  runs 目录：%s" % RUNS)
    if a.open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


def main():
    ap = argparse.ArgumentParser(description="实验场观测层（只读）")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="本地观测台（默认）")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action="store_true", default=True)
    s.add_argument("--no-open", dest="open", action="store_false")
    l = sub.add_parser("ls", help="列出所有 run")
    l.add_argument("--kind", default="key", choices=["key", "dev", "all"])
    sh = sub.add_parser("show", help="终端时间线")
    sh.add_argument("tag")
    sh.add_argument("--full", type=int, default=0, help="附带原始观察的截断长度")
    ex = sub.add_parser("export", help="导出单文件 HTML（双击即看，无需服务）")
    ex.add_argument("tag")
    ex.add_argument("--out", help="输出路径，默认 runs/<tag>/viewer.html")
    ex.add_argument("--inline", action="store_true", help="把帧也内嵌进 HTML（单个文件，体积大但可随处带走）")
    ex.add_argument("--open", action="store_true", default=True)
    ex.add_argument("--no-open", dest="open", action="store_false")
    a = ap.parse_args()
    if a.cmd == "ls":
        cmd_ls(a)
    elif a.cmd == "show":
        cmd_show(a)
    elif a.cmd == "export":
        cmd_export(a)
    else:
        if a.cmd is None:
            a = ap.parse_args(["serve"])
        cmd_serve(a)


if __name__ == "__main__":
    main()
