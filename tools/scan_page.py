"""掃試聽版裡的口癖、連續重複的詞、過長的空白，輸出可以逐項點播的 掃描結果.html。

用法：python tools/scan_page.py <集資料夾>
要先跑過 render_preview.py（用它輸出的 work/pieces.json 和 work/試聽版_raw.wav）。
work/scan_notes.json（選用）：{"R:原始秒數取整": "備註"}，會顯示在那一列的「判斷」欄。
"""
import collections
import html
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

TICS = ["就是", "我覺得", "這個", "然後", "因為", "什麼", "其實", "所以", "可能", "好像", "那個", "比如說", "真的", "那種"]
CLUSTER_N, CLUSTER_WIN = 3, 8.0   # 同一個人 8 秒內同一個口癖出現 3 次以上
GAP_DB, GAP_MIN = -45, 1.0        # 試聽版裡低於 -45 dB 超過 1 秒
BACKCHANNEL = set("對嗯哦喔哈啊欸好")


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main(ep):
    ep = Path(ep)
    segs = json.loads((ep / "work/segments.json").read_text())
    pieces = json.loads((ep / "work/pieces.json").read_text())
    notes_f = ep / "work/scan_notes.json"
    notes = json.loads(notes_f.read_text()) if notes_f.exists() else {}

    def out_t(t):
        for p in pieces:
            if p["src"][0] - 1e-6 <= t < p["src"][1]:
                return p["out"] + t - p["src"][0]

    def src_t(o):
        for p in pieces:
            if p["out"] <= o < p["out"] + p["src"][1] - p["src"][0]:
                return p["src"][0] + o - p["out"]

    chars = []  # 保留下來的每個字：(字, 試聽版時間, 原始時間, 說話者)
    for g in segs:
        for w in g["words"]:
            o = out_t(w["start"])
            if o is not None:
                chars += [(c, o, w["start"], g["speaker"]) for c in w["word"].strip()]
    chars.sort(key=lambda c: c[1])
    txt = "".join(c[0] for c in chars)
    total = chars[-1][1]

    def row(kind, a, b, extra=""):
        """a、b 是 txt 的字元範圍。"""
        lo, hi = max(0, a - 10), min(len(txt), b + 10)
        t0, t1 = chars[a][1], chars[min(b, len(chars) - 1)][1]
        key = f"{kind}:{int(chars[a][2])}"
        return dict(kind=kind, t=t0, play=(max(0, t0 - 3), t1 + 3.5), who=chars[a][3], src=chars[a][2],
                    text=html.escape(txt[lo:a]) + "<mark>" + html.escape(txt[a:b]) + "</mark>" + html.escape(txt[b:hi]),
                    extra=extra, note=notes.get(key, ""), key=key)

    # 1. 口癖：總數，以及密集出現的地方
    counts, clusters = [], []
    for f in TICS:
        occ = [m.start() for m in re.finditer(re.escape(f), txt)]
        by = collections.Counter(chars[a][3] for a in occ)
        counts.append((f, len(occ), by, len(occ) / total * 60))
        for who in by:
            mine = [a for a in occ if chars[a][3] == who]
            i = 0
            while i < len(mine):
                j = i
                while j + 1 < len(mine) and chars[mine[j + 1]][1] - chars[mine[i]][1] <= CLUSTER_WIN:
                    j += 1
                if j - i + 1 >= CLUSTER_N:
                    clusters.append(row("K", mine[i], mine[j] + len(f), f"「{f}」{j - i + 1} 次"))
                    i = j + 1
                else:
                    i += 1
    clusters.sort(key=lambda r: r["t"])

    # 2. 連續重複的詞
    repeats = []
    for m in re.finditer(r"(.{1,4}?)\1+", txt):
        s, u = m.group(0), m.group(1)
        if set(s) <= BACKCHANNEL or (len(s) < 4 and u not in "我你他"):
            continue
        repeats.append(row("R", m.start(), m.end()))

    # 3. 過長的空白
    with wave.open(str(ep / "work/試聽版_raw.wav")) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    hop = int(sr * 0.01)
    n = len(x) // hop
    quiet = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9) < GAP_DB
    gaps, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 0.01 >= GAP_MIN:
                t, d = i * 0.01, (j - i) * 0.01
                s = src_t(t + d / 2)
                before = "".join(c[0] for c in chars if t - 5 <= c[1] < t + d / 2)[-12:]
                after = "".join(c[0] for c in chars if t + d / 2 <= c[1] < t + d + 5)[:12]
                key = f"P:{int(s) if s else 0}"
                gaps.append(dict(kind="P", t=t, play=(max(0, t - 3), t + d + 3), who="", src=s or 0,
                                 text=html.escape(before) + f"<mark> ‖ {d:.1f} 秒 ‖ </mark>" + html.escape(after),
                                 extra=f"{d:.1f} 秒", note=notes.get(key, ""), key=key, d=d))
            i = j
        else:
            i += 1

    def table(rows, prefix):
        out = ["<table><tr><th>編號</th><th>試聽版</th><th>誰</th><th>內容</th><th></th><th>判斷</th></tr>"]
        for k, r in enumerate(rows, 1):
            out.append(f"<tr><td>{prefix}{k}</td><td>{mmss(r['t'])}</td><td>{r['who']}</td>"
                       f"<td>{r['text']}{' <small>' + r['extra'] + '</small>' if r['extra'] else ''}</td>"
                       f"<td><button onclick=\"play({r['play'][0]:.2f},{r['play'][1]:.2f},this)\">▶</button></td>"
                       f"<td>{html.escape(r['note'])}</td></tr>")
        return "\n".join(out + ["</table>"])

    who = sorted({c[3] for c in chars})
    cnt = ["<table><tr><th>口癖</th><th>總次數</th>" + "".join(f"<th>{w_}</th>" for w_ in who) + "<th>每分鐘</th></tr>"]
    for f, tot, by, rate in sorted(counts, key=lambda c: -c[1]):
        cnt.append(f"<tr><td>{f}</td><td>{tot}</td>" + "".join(f"<td>{by.get(w_, 0)}</td>" for w_ in who) + f"<td>{rate:.1f}</td></tr>")
    cnt.append("</table>")

    page = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>掃描結果</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--line:#ddd;--mark:#ffe58a;--dim:#666}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e6e6e6;--line:#333;--mark:#6b5a00;--dim:#999}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"PingFang TC",sans-serif;max-width:980px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 28px}}td,th{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}}
mark{{background:var(--mark);color:inherit}}small{{color:var(--dim)}}button{{font-size:15px;cursor:pointer}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}}
</style>
<div class="bar"><audio id="a" src="試聽版.mp3" preload="metadata" controls style="width:100%"></audio></div>
<h1>掃描結果</h1>
<p>試聽版 {mmss(total)}。按 ▶ 會從那個位置前 3 秒播到後 3 秒。回饋時講編號，例如「R3、R7 剪，P2 縮」。</p>
<h2>口癖總數</h2>{''.join(cnt)}
<h2>K　口癖密集處（同一人 {CLUSTER_WIN:.0f} 秒內 {CLUSTER_N} 次以上，共 {len(clusters)} 處）</h2>{table(clusters, 'K')}
<h2>R　連續重複的詞（共 {len(repeats)} 處）</h2>{table(repeats, 'R')}
<h2>P　超過 {GAP_MIN:.0f} 秒的空白（共 {len(gaps)} 處）</h2>{table(gaps, 'P')}
<script>
const a=document.getElementById('a');let stop=null;
a.addEventListener('timeupdate',()=>{{if(stop!==null&&a.currentTime>=stop){{a.pause();stop=null}}}});
function play(s,e){{stop=e;a.currentTime=s;a.play()}}
</script></html>"""
    (ep / "掃描結果.html").write_text(page)
    dump = [dict(id=f"{p}{k}", key=r["key"], t=mmss(r["t"]), who=r["who"], src=round(r["src"], 2),
                 text=re.sub("<[^>]+>", "|", r["text"]), extra=r["extra"])
            for p, rows in (("K", clusters), ("R", repeats), ("P", gaps)) for k, r in enumerate(rows, 1)]
    (ep / "work/scan_items.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1))
    print(f"已輸出 掃描結果.html：K {len(clusters)}、R {len(repeats)}、P {len(gaps)}")


if __name__ == "__main__":
    main(sys.argv[1])
