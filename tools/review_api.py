"""逐字稿頁（/review）背後的邏輯：每個字現在是留是刪、圈字標刪、改回留著、把一段「剪後」當場拼出來。

serve.py 呼叫這裡的函式；頁面是 tools/review.html。
"""
import bisect
import json
import wave

import numpy as np

import render_preview as rp

QUIET_DB = 12      # 比整集底噪高不到這麼多算停頓
QUIET_MIN = 0.04   # 停頓至少這麼長才拿來當刀點
REACH = 0.3        # 圈字的起訖往前後找停頓的範圍
MIN_LEFT = 0.05    # 改回留著之後，剩下不到這麼長的刪除片段直接丟掉


def sec(label):
    p = [int(x) for x in label.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


class Review:
    """一集裡不會變的東西：章節、每個字的時間。edl 每次從檔案重讀。"""

    def __init__(self, ep_dir):
        self.ep = rp.Episode(ep_dir)
        self.dir = self.ep.ep
        chapters = json.loads((self.dir / "sections.json").read_text())
        segs = self.ep.segments
        end = max(s["end"] for s in segs) + 1
        flat = [s for c in chapters for s in c["sections"]]
        starts = sorted(s["start"] for s in segs)
        # 段落邊界用那一秒開始的第一句的實際時間，避免整數秒切出零碎片段
        bounds = [next((t for t in starts if int(t) >= sec(s["start"])), sec(s["start"])) for s in flat] + [end]
        bounds[0] = 0.0
        self.words = []   # (start, end, 字)，順序和頁面上一樣
        for i, s in enumerate(flat):
            s["a"], s["b"], s["lines"] = bounds[i], bounds[i + 1], []
            for g in segs:
                if s["a"] <= g["start"] < s["b"]:
                    s["lines"].append({"t": round(g["start"], 2), "sp": g["speaker"], "n": len(g["words"])})
                    self.words += [(w["start"], w["end"], w["word"]) for w in g["words"]]
        self.chapters = chapters
        self.floor = float(np.percentile(self.ep.loud, 5))

    def edl(self):
        return json.loads((self.dir / "edl.json").read_text())

    def write(self, edl):
        rp.plan(self.ep, edl)  # 存之前先確認整份 edl 還排得出來
        (self.dir / "edl.json").write_text(json.dumps(edl, ensure_ascii=False, indent=1))

    # ---- 給頁面的資料 ----

    def static(self):
        cfg = self.ep.cfg
        return dict(title=cfg["title"], limit=cfg["limit_min"] * 60, speakers=[t["speaker"] for t in cfg["tracks"]],
                    chapters=self.chapters, words=[[w, round(s, 2), round(e, 2)] for s, e, w in self.words],
                    state=self.state())

    def state(self, edl=None):
        """每個字的狀態：-2 留著；-1 剪掉但頁面改不了（搬移的接點、頭尾）；0 以上是 remove 的編號。"""
        edl = edl or self.edl()
        blocks, pieces, fades = rp.plan(self.ep, edl)
        by_src = sorted(pieces)
        starts = [p[0] for p in by_src]
        live = [(j, r["from"], r["to"]) for j, r in enumerate(edl["remove"])
                if any(b["a"] < r["from"] and r["to"] < b["b"] for b in blocks)]
        status = []
        for s, e, _ in self.words:
            t = (s + e) / 2
            i = bisect.bisect_right(starts, t) - 1
            if i >= 0 and t < by_src[i][1]:
                status.append(-2)
            else:
                # 被搬走的段落，原位置還蓋著一刀大的；算在包住這個字的最小那一刀頭上
                status.append(min(((b - a, j) for j, a, b in live if a <= t < b), default=(0, -1))[1])
        total = sum(b - a for a, b in pieces) - sum(fades[1:])
        return dict(status=status, total=round(total, 2),
                    pieces=[[round(a, 3), round(b, 3), f] for (a, b), f in zip(pieces, fades)],
                    removes=[dict(a=r["from"], b=r["to"], by=r.get("by", ""), kind=r.get("kind", ""), note=r.get("note", ""))
                             for r in edl["remove"]],
                    keeps=[[k["from"], k["to"]] for k in edl.get("keep", [])])

    # ---- 圈字 ----

    def quiet_near(self, t, lo_t, hi_t):
        """t 附近 REACH 秒內最近的停頓的中點，限制在 lo_t 到 hi_t 之間（不越過隔壁的字）；沒有停頓就回傳 t。"""
        hop = self.ep.hop
        lo = max(int(max(t - REACH, lo_t) / hop), 0)
        q = self.ep.loud[lo:int(min(t + REACH, hi_t) / hop) + 1] < self.floor + QUIET_DB
        best, k = None, 0
        while k < len(q):
            if q[k]:
                j = k
                while j < len(q) and q[j]:
                    j += 1
                if (j - k) * hop >= QUIET_MIN:
                    mid = (lo + (k + j) / 2) * hop
                    if best is None or abs(mid - t) < abs(best - t):
                        best = mid
                k = j
            else:
                k += 1
        return round(t if best is None else best, 3)

    def span(self, first, last):
        if not 0 <= first <= last < len(self.words):
            raise ValueError("圈選的範圍不對")
        mid = lambda i: (self.words[i][0] + self.words[i][1]) / 2  # noqa: E731
        s0, e1 = self.words[first][0], self.words[last][1]
        a = self.quiet_near(s0, mid(first - 1) if first and mid(first - 1) < s0 else s0 - REACH, mid(first))
        b = self.quiet_near(e1, mid(last), mid(last + 1) if last + 1 < len(self.words) and mid(last + 1) > e1 else e1 + REACH)
        if b - a < 0.05:
            raise ValueError("圈到的範圍太短")
        text = "".join(w for _, _, w in self.words[first:last + 1])
        return a, b, text if len(text) <= 40 else text[:38] + "…"

    def mark(self, first, last):
        """把第 first 到 last 個字標成刪掉。碰到已有的 remove 就併成一刀。"""
        a, b, text = self.span(first, last)
        edl = self.edl()
        blocks, _, _ = rp.plan(self.ep, edl)
        # 這段話現在是從哪一個區塊播出來的；只和那個區塊裡的刀合併
        home = next((blk for blk in blocks if any(x < b and a < y for x, y in blk["pieces"])), None)
        if home is None:
            raise ValueError("這段已經整個剪掉了")
        hit = [r for r in edl["remove"] if r["from"] < b and a < r["to"] and home["a"] < r["from"] and r["to"] < home["b"]]
        lo, hi = min([a] + [r["from"] for r in hit]), max([b] + [r["to"] for r in hit])
        if not (home["a"] < lo and hi < home["b"]):
            raise ValueError("這段碰到搬移的接點或頭尾，頁面改不了，請在 session 裡說")
        if hit:
            new = dict(hit[0])
            new["note"] = "；".join([r.get("note", "") for r in hit] + [f"圈選「{text}」"])
            last_hit = max(hit, key=lambda r: r["to"])
            for k in ("fade", "then"):
                new.pop(k, None)
                if k in last_hit and last_hit["to"] == hi:   # 接點沒動才沿用它的淡化
                    new[k] = last_hit[k]
            if "then" in last_hit:
                new["then"] = last_hit["then"]
        else:
            new = {"note": f"圈選「{text}」", "by": "user", "init": [a, b]}
        new["from"], new["to"] = lo, hi
        edl["remove"] = sorted([r for r in edl["remove"] if r not in hit] + [new], key=lambda r: r["from"])
        edl["keep"] = [k for k in edl.get("keep", []) if not (k["from"] < hi and lo < k["to"])]
        self.write(edl)
        return f"圈選刪掉「{text}」{rp.mmss(a)}", self.state(edl)

    def keep(self, first=None, last=None, remove=None):
        """改回留著：給 remove 編號就整刀留；給字的範圍就只留那一段，原本的刀切成前後兩段。"""
        edl = self.edl()
        if remove is not None:
            r = edl["remove"][remove]
            a, b, text = r["from"], r["to"], r.get("note", "")[:40]
            hit = [r]
        else:
            a, b, text = self.span(first, last)
            st = self.state(edl)["status"][first:last + 1]
            hit = [edl["remove"][j] for j in sorted({j for j in st if j >= 0})]
        if not hit:
            raise ValueError("這段沒有可以改回來的刪除")
        rest = [r for r in edl["remove"] if r not in hit]
        for r in hit:
            if a - r["from"] >= MIN_LEFT:      # 前半還是刪掉；接點是新的，不沿用原本的淡化
                rest.append({k: v for k, v in r.items() if k not in ("fade", "then", "init")} | {"to": a})
            if r["to"] - b >= MIN_LEFT:
                rest.append({k: v for k, v in r.items() if k != "init"} | {"from": b})
        edl["remove"] = sorted(rest, key=lambda r: r["from"])
        lo, hi = max(a, min(r["from"] for r in hit)), min(b, max(r["to"] for r in hit))
        edl.setdefault("keep", []).append({"from": lo, "to": hi, "note": f"改回留著：{text}",
                                           "was": "；".join(r.get("note", "") for r in hit)[:200]})
        self.write(edl)
        return f"改回留著「{text}」{rp.mmss(lo)}", self.state(edl)

    # ---- 當場拼出剪後 ----

    def preview(self, pieces):
        """pieces：[(原始起點, 原始終點, 和前一段的淡化秒數)]，照播出順序。回傳 mono 16-bit WAV 的 bytes。"""
        if not pieces or any(b <= a for a, b, _ in pieces) or sum(b - a for a, b, _ in pieces) > 900:
            raise ValueError("片段不對，或總長超過 15 分鐘")
        mix = _LazyMix(self.dir, self.ep.cfg["tracks"])
        out = rp.render(mix, mix.sr, [(a, b) for a, b, _ in pieces], [f for _, _, f in pieces])
        rms = float(np.sqrt((out ** 2).mean())) + 1e-9
        out = np.clip(out * min(8.0, 0.1 / rms), -1, 1)   # 原始混音偏小聲，拉到大約 -20 dBFS
        data = (out * 32767).astype(np.int16).tobytes()
        import struct
        return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, mix.sr, mix.sr * 2, 2, 16) + b"data" + struct.pack("<I", len(data)) + data)


class _LazyMix:
    """讓 rp.render 可以只讀需要的片段，不用把整集載進記憶體。"""

    def __init__(self, ep_dir, tracks):
        self.files = [(wave.open(str(ep_dir / t["file"])), 10 ** (t["gain_db"] / 20)) for t in tracks]
        self.sr = self.files[0][0].getframerate()

    def __getitem__(self, sl):
        out = None
        for w, gain in self.files:
            w.setpos(min(max(sl.start, 0), w.getnframes()))
            x = np.frombuffer(w.readframes(max(sl.stop - sl.start, 0)), dtype=np.int16).astype(np.float32) / 32768 * gain
            out = x if out is None else out[:min(len(out), len(x))] + x[:min(len(out), len(x))]
        return out
