"""依 edl.json 把對話軌剪成試聽版，並寫出刀點檢查與刪減清單。

用法：python tools/render_preview.py <集資料夾> [--check]
--check 只印出每個刀點前後的字，不輸出音檔。
需要：edl.json、work/segments.json、work/envelope.npz、work/pauses.json
輸出：試聽版.mp3、刪減清單.md
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import config

XFADE = 0.015     # 每個接點的交叉淡化
SNAP_BEFORE = 0.3  # 刀點往前找最安靜處的範圍
SNAP_AFTER = 0.1
PAUSE_KEEP = 0.6


def hms(t):
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def mmss(t):
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


class Episode:
    def __init__(self, ep):
        self.ep = Path(ep)
        self.cfg = config.load(self.ep)
        self.segments = json.loads((self.ep / "work/segments.json").read_text())
        self.words = [w for s in self.segments for w in s["words"]]
        env = np.load(self.ep / "work/envelope.npz")
        self.hop = float(env["hop"])
        self.loud = np.max([env[t["key"]] + t["gain_db"] for t in self.cfg["tracks"]], axis=0)
        self.pauses = json.loads((self.ep / "work/pauses.json").read_text())["pauses"]

    def resolve(self, spec):
        """把 edl 的時間寫法換成秒。"""
        if isinstance(spec, (int, float)):
            return float(spec)
        label, _, phrase = spec.partition("|")
        segs = [s for s in self.segments if hms(s["start"]) == label]
        if not segs:
            raise ValueError(f"找不到 {label} 開始的段落")
        if not phrase:
            return min(s["start"] for s in segs)
        for s in segs:
            text, starts = "", []
            for w in s["words"]:
                starts += [w["start"]] * len(w["word"])
                text += w["word"]
            i = text.find(phrase)
            if i >= 0:
                return starts[i]
        raise ValueError(f"{label} 的段落裡找不到「{phrase}」")

    def snap(self, t):
        """移到附近兩軌都最安靜的位置，避免切到字。"""
        a = max(int((t - SNAP_BEFORE) / self.hop), 0)
        b = int((t + SNAP_AFTER) / self.hop)
        return (a + int(np.argmin(self.loud[a:b]))) * self.hop

    def context(self, t, before):
        ws = [w for w in self.words if (w["end"] <= t + 0.05 if before else w["start"] >= t - 0.05)]
        ws = ws[-10:] if before else ws[:10]
        return "".join(w["word"] for w in ws)


def pieces_for(ep, a, b, removes):
    """一個保留區塊扣掉長停頓與要刪的字之後的片段。"""
    holes = [(s, e) for s, e in removes if a < s and e < b]
    for s, e in ep.pauses:
        # 碰到 remove 的停頓不縮：那個接點的位置是手動定的，不能被縮停頓移走
        if a < s and e < b and not any(s < re and rs < e for rs, re in removes):
            mid = (s + e) / 2
            holes.append((mid - (e - s - PAUSE_KEEP) / 2, mid + (e - s - PAUSE_KEEP) / 2))
    merged = []  # 重疊的洞併成一個，否則會算出長度為負的片段
    for s, e in sorted(holes):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out, cur = [], a
    for s, e in merged:
        out.append((cur, s))
        cur = e
    out.append((cur, b))
    return out


def plan(ep, edl):
    """把 edl 換成每一塊的秒數與片段，以及每個片段開頭那個接點的淡化秒數。

    淡化預設 XFADE。remove 項目的 "fade" 管它造成的那個接點；block 的 "fade" 管它和前一塊的接點。
    """
    removes = [(r["from"], r["to"]) for r in edl["remove"]]
    custom = {round(r["to"], 3): r["fade"] for r in edl["remove"] if "fade" in r}
    blocks = []
    for blk in edl["blocks"]:
        # 純數字是手動指定的刀點，不再對齊
        a, b = (ep.resolve(blk[k]) if isinstance(blk[k], (int, float)) else ep.snap(ep.resolve(blk[k]))
                for k in ("from", "to"))
        if "fade" in blk:
            custom[round(a, 3)] = blk["fade"]
        blocks.append(dict(blk, a=a, b=b, pieces=pieces_for(ep, a, b, removes)))
    pieces = [p for blk in blocks for p in blk["pieces"]]
    fades = [custom.get(round(a, 3), XFADE) for a, _ in pieces]
    return blocks, pieces, fades


def load_mix(ep):
    tracks = []
    for t in ep.cfg["tracks"]:
        with wave.open(str(ep.ep / t["file"])) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        tracks.append(x * 10 ** (t["gain_db"] / 20))
    n = min(len(x) for x in tracks)
    mix = tracks[0][:n]
    for x in tracks[1:]:
        mix = mix + x[:n]
    return mix, sr


def render(mix, sr, pieces, fades=None):
    """fades[i] 是第 i 個片段和前一個片段交叉淡化的秒數，沒給就都用 XFADE。"""
    chunks, tail = [], np.zeros(0, dtype=np.float32)  # tail 是前一個片段，還可能被下一個接點改到
    for i, (a, b) in enumerate(pieces):
        x = mix[int(a * sr):int(b * sr)].copy()
        fade = min(int((fades[i] if fades else XFADE) * sr), len(tail), len(x))
        if fade > 0:
            ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
            tail[-fade:] = tail[-fade:] * ramp[::-1] + x[:fade] * ramp
            x = x[fade:]
        chunks.append(tail)
        tail = x
    return np.concatenate(chunks + [tail])


def main(ep_dir, check_only):
    ep = Episode(ep_dir)
    edl = json.loads((ep.ep / "edl.json").read_text())
    blocks, all_pieces, fades = plan(ep, edl)

    print("刀點檢查（‖ 是接點）")
    for prev, nxt in zip(blocks, blocks[1:]):
        print(f"{mmss(prev['b'])} → {mmss(nxt['a'])}：…{ep.context(prev['b'], True)} ‖ {ep.context(nxt['a'], False)}…")

    total = sum(b - a for a, b in all_pieces) - sum(fades[1:])
    print(f"試聽版長度約 {mmss(total)}")
    if check_only:
        return

    # 刪減清單：每一塊在試聽版的位置，以及它後面剪掉了什麼
    lines = [f"# 刪減清單：{ep.cfg['title']} 試聽版", "",
             f"試聽版長度 {mmss(total)}，原始 {mmss(len(ep.loud) * ep.hop)}。只有對話，還沒混音、沒有片頭和片尾曲。",
             "左欄是試聽版的時間，聽的時候對照這欄。要救回或再砍，直接寫「試聽版 12:30 那段要救回來」。", "",
             "| 試聽版 | 原始 | 保留 | 之後剪掉 |", "|---|---|---|---|"]
    t = 0.0
    for blk in blocks:
        # 區塊裡帶 then 的 remove 是整段刪，在那裡分列
        cuts = sorted((r for r in edl["remove"] if "then" in r and blk["a"] < r["from"] and r["to"] < blk["b"]), key=lambda r: r["from"])
        a, keep = blk["a"], blk["keep"]
        for r in cuts + [None]:
            b = r["from"] if r else blk["b"]
            lines.append(f"| {mmss(t)} | {mmss(a)}–{mmss(b)} | {keep} | {r['note'] if r else blk['cut_after']} |")
            t += sum(min(y, b) - max(x, a) for x, y in blk["pieces"] if min(y, b) > max(x, a))
            if r:
                a, keep = r["to"], r["then"]
    lines += ["", "另外縮短了保留區塊裡所有人同時沉默超過 1.2 秒的停頓（縮成 0.6 秒），並刪掉 edl.json `remove` 裡沒有 then 的小片段（填充詞與逐章回饋）。"]
    (ep.ep / "刪減清單.md").write_text("\n".join(lines) + "\n")

    # 每個片段在試聽版裡的位置，給對照頁用。接點的交叉淡化會讓後面的片段往前疊。
    pieces, t = [], 0.0
    for i, (a, b) in enumerate(all_pieces):
        if i:
            t -= fades[i]
        pieces.append({"src": [round(a, 3), round(b, 3)], "out": round(t, 3)})
        t += b - a
    (ep.ep / "work/pieces.json").write_text(json.dumps(pieces))

    mix, sr = load_mix(ep)
    out = render(mix, sr, all_pieces, fades)
    raw = ep.ep / "work/試聽版_raw.wav"
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-nostats", "-loglevel", "error", "-y", "-i", str(raw),
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-b:a", "128k",
                    str(ep.ep / "試聽版.mp3")], check=True)
    print("已輸出 試聽版.mp3、刪減清單.md")


if __name__ == "__main__":
    main(sys.argv[1], "--check" in sys.argv)
