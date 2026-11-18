"""在逐字稿裡找一句話，印出它在原始錄音的秒數，方便寫進 edl.json。

用法：python tools/find.py <集資料夾> <片語>
片語可以跨說話者，不含標點。每個結果印出：起訖秒數、誰說的、前後文、現在是留著還是已經剪掉。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import render_preview as rp  # noqa: E402


def main(ep_dir, phrase):
    ep = rp.Episode(ep_dir)
    _, pieces, _ = rp.plan(ep, json.loads((ep.ep / "edl.json").read_text()))
    chars = [(c, w, s["speaker"]) for s in ep.segments for w in s["words"] for c in w["word"]]
    text = "".join(c for c, _, _ in chars)
    i, found = text.find(phrase), 0
    while i >= 0:
        first, last = chars[i][1], chars[i + len(phrase) - 1][1]
        mid = (first["start"] + last["end"]) / 2
        kept = any(a <= mid < b for a, b in pieces)
        who = "、".join(dict.fromkeys(sp for _, _, sp in chars[i:i + len(phrase)]))
        print(f"{first['start']:.2f}–{last['end']:.2f}（{rp.hms(first['start'])}）{who}　{'留著' if kept else '已剪掉'}　"
              f"…{text[max(0, i - 12):i]}【{phrase}】{text[i + len(phrase):i + len(phrase) + 12]}…")
        found += 1
        i = text.find(phrase, i + 1)
    if not found:
        print("找不到。逐字稿沒有標點，也可能有錯字，試試短一點的片語。")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
