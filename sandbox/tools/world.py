# -*- coding: utf-8 -*-
"""world.py — 唯一入口：把任意格式读成「一组世界坐标三角网格实体」。

设计约束（很重要，别加东西）：
  * 只输出几何：顶点数组 + 三角面索引。名称仅在 --keep-names 时保留，用于人工对拍。
  * 单位与朝向在这里裁决，并把裁决依据写进 meta，供审计。
  * 不做任何分类、识别、命名猜测 —— 那是 plan 与 LLM 的活。
"""
import json
import math
import os
import re
import urllib.parse

import numpy as np

# ---------------------------------------------------------------- 实体


class Solid:
    __slots__ = ("sid", "name", "P", "F", "src")

    def __init__(self, sid, name, P, F, src="?"):
        self.sid = sid
        self.name = name
        self.P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        self.F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
        self.src = src

    def __repr__(self):
        return f"<Solid {self.sid} {self.name} tri={len(self.F)}>"

    def bbox(self):
        return self.P.min(0), self.P.max(0)


# ---------------------------------------------------------------- STEP / IFC

_STEP_ENT = re.compile(r"#(\d+)\s*=\s*(.*?);\s*\r?\n", re.S)
_NUM = r"(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
_TRI = re.compile(r"\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)")
_IDX = re.compile(r"\(([\d,]+)\)")


def _step_entities(text):
    return {int(m.group(1)): m.group(2).strip() for m in _STEP_ENT.finditer(text)}


def _refs(ent, eid):
    return [int(x) for x in re.findall(r"#(\d+)", ent.get(eid, ""))]


def _plist(ent, eid):
    return [tuple(map(float, t)) for t in _TRI.findall(ent[eid])]


def _faceset(ent, eid):
    e = ent[eid]
    tgt = int(re.match(r"IFCTRIANGULATEDFACESET\(#(\d+)", e).group(1))
    body = e[e.index(".T.,") + 4:] if ".T.," in e else e
    return tgt, [tuple(int(x) - 1 for x in f.split(",")) for f in _IDX.findall(body)]


def _axis2placement(ent, eid):
    r = _refs(ent, eid)
    loc = _plist(ent, r[0])[0] if r and ent.get(r[0], "").startswith("IFCCARTESIANPOINT") else (0, 0, 0)
    z = np.array(_plist(ent, r[1])[0], float) if len(r) > 1 and ent.get(r[1], "").startswith("IFCDIRECTION") else np.array([0, 0, 1.0])
    x = np.array(_plist(ent, r[2])[0], float) if len(r) > 2 and ent.get(r[2], "").startswith("IFCDIRECTION") else np.array([1.0, 0, 0])
    z = z / np.linalg.norm(z)
    x = x - z * x.dot(z)
    x = x / np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z]), np.array(loc, float)


def _local_placement(ent, eid, depth=0):
    if eid is None or depth > 24:
        return np.eye(3), np.zeros(3)
    ax, par = None, None
    for rv in _refs(ent, eid):
        t = ent.get(rv, "")
        if t.startswith("IFCAXIS2PLACEMENT3D"):
            ax = _axis2placement(ent, rv)
        elif t.startswith("IFCLOCALPLACEMENT"):
            par = rv
    if ax is None:
        ax = (np.eye(3), np.zeros(3))
    if par is not None:
        pR, pt = _local_placement(ent, par, depth + 1)
        ax = (pR @ ax[0], pt + pR @ ax[1])
    return ax


_IFC_PROD = re.compile(
    r"IFC(WALL|WALLSTANDARDCASE|SLAB|DOOR|WINDOW|FURNITURE|BUILDINGELEMENTPROXY|"
    r"OPENINGELEMENT|BUILDINGELEMENTPART|PLATE|MEMBER|COLUMN|BEAM|COVERING|SPACE|"
    r"STAIR|RAILING|ROOF|CURTAINWALL|SITE|BUILDING|BUILDINGSTOREY)\("
)
_IFC_NONGEOM = {"SPACE", "SITE", "BUILDING", "BUILDINGSTOREY"}


def _ifc_unit(ent):
    """从 IFCSIUNIT 读长度单位；读不到就返回 None（交给几何尺度推断）。"""
    for v in ent.values():
        if v.startswith("IFCSIUNIT") and ".LENGTHUNIT." in v:
            pre = re.search(r"\.(EXA|PETA|TERA|GIGA|MEGA|KILO|HECTO|DECA|DECI|CENTI|MILLI|MICRO|NANO)\.", v)
            if pre:
                return {"MILLI": 1e-3, "CENTI": 1e-2, "DECI": 1e-1, "MICRO": 1e-6}.get(pre.group(1), 1.0), f"IFCSIUNIT .{pre.group(1)}."
            return 1.0, "IFCSIUNIT no prefix"
        if v.startswith("IFCCONVERSIONBASEDUNIT") and "LENGTHUNIT" in v:
            r = _refs(ent, list(ent.keys())[list(ent.values()).index(v)])
            for rid in r:
                if ent.get(rid, "").startswith("IFCMEASUREWITHUNIT"):
                    mm = re.search(r"IFCMEASUREWITHUNIT\(IFCLENGTHMEASURE\(([-\d.Ee+]+)\)", ent[rid])
                    if mm:
                        return float(mm.group(1)), "IFCCONVERSIONBASEDUNIT"
    return None


def load_ifc(path, keep_names=False):
    text = open(path, encoding="utf-8", errors="ignore").read()
    ent = _step_entities(text)
    unit, unit_src = _ifc_unit(ent)
    solids = []
    for eid, body in ent.items():
        m = _IFC_PROD.match(body)
        if not m or m.group(1) in _IFC_NONGEOM:
            continue
        refs = _refs(ent, eid)
        shape = [r for r in refs if ent.get(r, "").startswith("IFCPRODUCTDEFINITIONSHAPE")]
        if not shape:
            continue
        pl = [r for r in refs if ent.get(r, "").startswith("IFCLOCALPLACEMENT")]
        R, t = _local_placement(ent, pl[0]) if pl else (np.eye(3), np.zeros(3))
        P, F = [], []
        for sp in shape:
            for r in _refs(ent, sp):
                if not ent.get(r, "").startswith("IFCSHAPEREPRESENTATION"):
                    continue
                for f in _refs(ent, r):
                    if not ent.get(f, "").startswith("IFCTRIANGULATEDFACESET"):
                        continue
                    pid, fs = _faceset(ent, f)
                    off = len(P)
                    P.extend((R @ np.array(q, float) + t).tolist() for q in _plist(ent, pid))
                    F.extend((a + off, b + off, c + off) for a, b, c in fs)
        if not F:
            continue
        nm = re.search(r"\$,\s*'([^']*)'", body)
        solids.append(Solid(f"C{len(solids):03d}", nm.group(1) if (nm and keep_names) else "s", P, F, m.group(1)))
    meta = {"format": "ifc", "unit": unit, "unit_source": unit_src, "header_unit": unit is not None}
    return solids, meta


# ---------------------------------------------------------------- OBJ


def load_obj(path, keep_names=False):
    verts, groups = [], []
    cur = None
    for ln in open(path, encoding="utf-8", errors="ignore"):
        if ln.startswith("v "):
            verts.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("o ") or ln.startswith("g "):
            cur = {"name": ln.split(None, 1)[1].strip(), "F": []}
            groups.append(cur)
        elif ln.startswith("f "):
            if cur is None:
                cur = {"name": "default", "F": []}
                groups.append(cur)
            idx = []
            for tok in ln.split()[1:]:
                idx.append(int(tok.split("/")[0]) - 1)
            for k in range(1, len(idx) - 1):
                cur["F"].append((idx[0], idx[k], idx[k + 1]))
    V = np.array(verts, float)
    solids = []
    for g in groups:
        if not g["F"]:
            continue
        f = np.array(g["F"], int)
        used = np.unique(f)
        remap = -np.ones(len(V), int)
        remap[used] = np.arange(len(used))
        solids.append(Solid(f"O{len(solids):03d}", g["name"] if keep_names else "s", V[used], remap[f], "OBJ"))
    return solids, {"format": "obj", "unit": None, "unit_source": "inferred", "header_unit": False}


# ---------------------------------------------------------------- glTF


def load_gltf(path, keep_names=False):
    g = json.load(open(path, encoding="utf-8"))
    base = os.path.dirname(path)
    bin_cache = {}

    def buf(i):
        if i not in bin_cache:
            uri = g["buffers"][i].get("uri") or ""
            if uri.startswith("data:"):
                # 单文件 glTF：buffer 以 base64 内嵌。编辑落盘默认走这条路，
                # 免得一个模型散成 .gltf + .bin 两个文件、拷走时丢一半。
                import base64
                bin_cache[i] = base64.b64decode(uri.split(",", 1)[1])
            elif uri:
                bin_cache[i] = open(os.path.join(base, urllib.parse.unquote(uri)), "rb").read()
            else:
                raise ValueError("glTF 的 buffer %d 没有 uri（GLB 内嵌块未支持）" % i)
        return bin_cache[i]

    CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

    def acc(i):
        a = g["accessors"][i]
        bv = g["bufferViews"][a["bufferView"]]
        data = buf(bv["buffer"])
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt = np.dtype(CT[a["componentType"]])
        n = a["count"] * NC[a["type"]]
        stride = bv.get("byteStride")
        if stride and stride != dt.itemsize * NC[a["type"]]:
            rows = []
            for k in range(a["count"]):
                s = off + k * stride
                rows.append(np.frombuffer(data, dt, NC[a["type"]], s))
            arr = np.vstack(rows)
        else:
            arr = np.frombuffer(data, dt, n, off)
        return arr.reshape(a["count"], NC[a["type"]]) if NC[a["type"]] > 1 else arr

    solids = []
    for node in g.get("nodes", []):
        if "mesh" not in node:
            continue
        P, F = [], []
        for prim in g["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            p = np.array(acc(prim["attributes"]["POSITION"]), float)
            if "indices" in prim:
                ix = np.array(acc(prim["indices"]), int).reshape(-1, 3)
            else:
                ix = np.arange(len(p)).reshape(-1, 3)
            off = len(P)
            P.extend(p.tolist())
            F.extend((ix + off).tolist())
        if not F:
            continue
        if "matrix" in node:
            M = np.array(node["matrix"], float).reshape(4, 4).T
            P = (np.array(P) @ M[:3, :3].T + M[:3, 3]).tolist()
        elif any(k in node for k in ("translation", "rotation", "scale")):
            t = np.array(node.get("translation", [0, 0, 0]), float)
            s = np.array(node.get("scale", [1, 1, 1]), float)
            q = node.get("rotation", [0, 0, 0, 1])
            x, y, z, w = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            P = ((np.array(P) * s) @ R.T + t).tolist()
        solids.append(Solid(f"G{len(solids):03d}", node.get("name", "s") if keep_names else "s", P, F, "glTF"))
    # 单位：glTF 规范单位是米，但很多导出器写成毫米或干脆不写。优先读显式声明
    # （asset.extras.unit），读不到再交给 decide_scale 按场景尺度推断。
    # 这一条对 EDIT 闭环是硬要求：编辑落盘的文件若不声明单位，读回时会重新推断，
    # 一次「加宽 1.2 m」可能被读成 1200 m，闭环就假了。
    ex = (g.get("asset") or {}).get("extras") or {}
    _u = str(ex.get("unit", "")).strip().lower()
    _UMAP = {"metre": 1.0, "meter": 1.0, "m": 1.0, "metres": 1.0, "meters": 1.0,
             "millimetre": 1e-3, "millimeter": 1e-3, "mm": 1e-3,
             "centimetre": 1e-2, "centimeter": 1e-2, "cm": 1e-2,
             "inch": 0.0254, "foot": 0.3048, "feet": 0.3048}
    if _u in _UMAP:
        meta = {"format": "gltf", "unit": _UMAP[_u],
                "unit_source": "glTF asset.extras.unit = %s" % _u, "header_unit": True}
    else:
        meta = {"format": "gltf", "unit": None, "unit_source": "inferred", "header_unit": False}
    return solids, meta


# ---------------------------------------------------------------- USD


def load_usd(path, keep_names=False):
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(path)
    solids = []
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        fvc = m.GetFaceVertexCountsAttr().Get()
        fvi = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not fvc:
            continue
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        M = np.array(xf).T
        P = (np.array(pts, float) @ M[:3, :3].T + M[:3, 3])
        F, k = [], 0
        for c in fvc:
            idx = list(fvi[k:k + c])
            k += c
            for j in range(1, c - 1):
                F.append((idx[0], idx[j], idx[j + 1]))
        if F:
            solids.append(Solid(f"U{len(solids):03d}", prim.GetName() if keep_names else "s", P, F, "USD"))
    return solids, {"format": "usd", "unit": None, "unit_source": "inferred", "header_unit": False}


# ---------------------------------------------------------------- 单位裁决


def decide_scale(solids, meta, target_height_hint=3.0):
    """单位裁决：优先用文件声明；否则按『房间高度应在 2~5 m』这类弱先验推断。
    返回 scale 与依据。这一步必须留痕，因为它是 Task 01 泛化性的关键。
    """
    if meta.get("unit"):
        return meta["unit"], meta["unit_source"]
    P = np.vstack([s.P for s in solids])
    if len(P) == 0:
        return 1.0, "empty"
    ext = P.max(0) - P.min(0)
    diag = float(np.linalg.norm(ext))
    for sc, name in ((1.0, "metre"), (1e-3, "millimetre"), (1e-2, "centimetre"), (0.0254, "inch"), (0.3048, "foot")):
        if 2.0 * 1.2 <= diag * sc <= 1e3:
            return sc, f"inferred: scene diagonal {diag:.3f}×{sc} = {diag*sc:.2f} m fits built-environment prior"
    return 1.0, f"fallback (diagonal {diag:.3f})"


def load(path, keep_names=False):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ifc":
        solids, meta = load_ifc(path, keep_names)
    elif ext == ".obj":
        solids, meta = load_obj(path, keep_names)
    elif ext in (".gltf", ".glb"):
        solids, meta = load_gltf(path, keep_names)
    elif ext in (".usd", ".usda", ".usdc"):
        solids, meta = load_usd(path, keep_names)
    else:
        raise ValueError(f"unsupported format {ext}")
    scale, src = decide_scale(solids, meta)
    if scale != 1.0:
        for s in solids:
            s.P = s.P * scale
    meta.update({"scale": scale, "scale_source": src, "n_solids": len(solids), "path": os.path.abspath(path)})
    return solids, meta
