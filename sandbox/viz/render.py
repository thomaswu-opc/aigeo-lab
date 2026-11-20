# -*- coding: utf-8 -*-
"""render.py — 把账本里的几何状态画成帧。无头（不依赖显卡），评委能复现。

为什么不用 Omniverse / 浏览器：演示的价值在于「同一个数能被人眼验一遍」，
而不是好看。正交投影 + 画家算法足够，且 100% 可复跑。
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _rot(yaw_deg, pitch_deg):
    a, b = math.radians(yaw_deg), math.radians(pitch_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return Rx @ Rz


def _shade(n, base, light=(0.35, -0.45, 0.82)):
    l = np.array(light, float)
    l /= np.linalg.norm(l)
    k = 0.45 + 0.55 * abs(float(np.dot(n, l)))
    return tuple(min(255, int(c * k)) for c in base)


def frame(solids, path, size=(1280, 720), yaw=38, pitch=26,
          highlight=None, points=None, lines=None, title="", caption="", hud=None,
          zoom=None, fit=None, bg=(248, 249, 251), fg=(28, 32, 40)):
    highlight = highlight or {}
    points = points or []
    lines = lines or []
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img, "RGBA")

    allP = np.vstack([s.P for s in solids]) if solids else np.zeros((1, 3))
    ref = np.asarray(fit, float) if fit is not None else allP
    center = np.asarray(zoom, float) if zoom is not None else ref.mean(0)
    R = _rot(yaw, pitch)
    Qr = (ref - center) @ R.T
    ext = Qr[:, :2].max(0) - Qr[:, :2].min(0)
    ext = np.maximum(ext, 1e-3)
    scale = min(size[0] * 0.80 / ext[0], size[1] * 0.72 / ext[1])
    # 以投影后的包围盒中心对齐，避免顶点分布不均导致的偏台
    q2 = np.stack([Qr[:, 0] * scale, -Qr[:, 1] * scale], 1)
    cx = (q2[:, 0].max() + q2[:, 0].min()) / 2
    cy = (q2[:, 1].max() + q2[:, 1].min()) / 2
    off = np.array([size[0] / 2 - cx, size[1] / 2 + 14 - cy])

    def proj(P):
        q = (np.asarray(P) - center) @ R.T
        return np.stack([q[:, 0] * scale + off[0], -q[:, 1] * scale + off[1]], 1), q[:, 2]

    tris = []
    for s in solids:
        A, B, C = s.P[s.F[:, 0]], s.P[s.F[:, 1]], s.P[s.F[:, 2]]
        n = np.cross(B - A, C - A)
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-12
        n[ok] /= ln[ok, None]
        p2, depth = proj(np.stack([A, B, C], 1).reshape(-1, 3))
        p2 = p2.reshape(-1, 3, 2)
        dz = depth.reshape(-1, 3).mean(1)
        base = highlight.get(s.sid, (176, 182, 192))
        alpha = 255 if s.sid in highlight else 190
        for i in range(len(p2)):
            tris.append((dz[i], p2[i], _shade(n[i], base), alpha))
    tris.sort(key=lambda t: t[0])

    for _, tri, col, alpha in tris:
        d.polygon([tuple(p) for p in tri], fill=col + (alpha,), outline=(255, 255, 255, 40))

    # 见证线与见证点
    for seg in lines:
        a2, _ = proj([seg[0]])
        b2, _ = proj([seg[1]])
        d.line([tuple(a2[0]), tuple(b2[0])], fill=(220, 38, 38, 255), width=3)
    for pt in points:
        c, _ = proj([pt["p"]])
        x, y = c[0]
        r = 7
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255), outline=(220, 38, 38, 255), width=3)
        if pt.get("label"):
            d.text((x + 11, y - 10), pt["label"], font=_font(17), fill=(180, 20, 20))

    # 顶部标题
    d.rectangle([0, 0, size[0], 74], fill=(255, 255, 255, 235))
    d.line([0, 74, size[0], 74], fill=(222, 226, 232, 255), width=1)
    d.text((22, 14), title, font=_font(24), fill=fg)
    d.text((22, 46), caption, font=_font(17), fill=(90, 98, 110))
    if hud:
        f = _font(16)
        y = size[1] - 26 * len(hud) - 12
        for row in hud:
            d.text((22, y), row, font=f, fill=(60, 66, 76))
            y += 26
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)
    return path


def encode(frames_glob, out_mp4, fps=1.6):
    import subprocess
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", frames_glob,
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "20", out_mp4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return {"ok": r.returncode == 0, "cmd": " ".join(cmd), "err": r.stderr[-500:] if r.returncode else ""}
