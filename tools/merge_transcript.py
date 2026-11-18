"""把混合軌的 whisper 結果標上說話者，整理成逐字稿，並列出可刪的填充詞。

各軌分開轉錄會因為串音重複出現同一句，所以改成轉錄混合軌，
再逐字比較各軌在那個字的音量，決定是誰說的。

用法：python tools/merge_transcript.py <集資料夾>
需要：episode.json、work/transcribe/dialogue_mix16k.json、work/envelope.npz
輸出：work/逐字稿.md、work/segments.json、work/fillers.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

FILLERS = {"嗯", "呃", "就是", "然後", "那個", "欸"}
GAP = 0.15     # 填充詞前後至少要有這麼長的靜音
SEARCH = 0.3   # whisper 時間戳誤差，往外找靜音的範圍
QUIET_DB = 12  # 比自己底噪高不到這麼多算靜音



def hms(t):
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def has_gap(quiet, a, b, g):
    """quiet[a:b] 裡有沒有連續 g 格的靜音。"""
    run = 0
    for q in quiet[max(a, 0):b]:
        run = run + 1 if q else 0
        if run >= g:
            return True
    return False


def main(ep):
    ep = Path(ep)
    env = np.load(ep / "work/envelope.npz")
    hop = float(env["hop"])
    cfg = config.load(ep)
    if cfg.get("language", "zh") == "zh":   # whisper 的中文常輸出簡體，轉成台灣繁體
        import opencc
        convert = opencc.OpenCC("s2tw").convert
    else:
        convert = str
    names = [t["speaker"] for t in cfg["tracks"]]
    level = [env[t["key"]] + t["gain_db"] for t in cfg["tracks"]]  # mic 比較小聲的人先補增益再比
    quiet = {name: x < np.percentile(x, 5) + QUIET_DB for name, x in zip(names, level)}
    data = json.loads((ep / "work/transcribe/dialogue_mix16k.json").read_text())

    words = []
    for seg in data["segments"]:
        for w in seg.get("words", []):
            a = int(w["start"] / hop)
            b = max(int(w["end"] / hop), a + 1)
            # 取那個字最大聲的部分比較，避開字邊緣的靜音
            peak = [np.percentile(x[a:b], 90) for x in level]
            top, second = sorted(peak, reverse=True)[:2]
            words.append({"start": w["start"], "end": w["end"], "word": convert(w["word"]).strip(),
                          "speaker": names[int(np.argmax(peak))], "margin": round(float(top - second), 1)})

    # whisper 的字級時間戳會偏，換人處常把一兩個字判給對方。
    # 夾在同一人中間、不超過兩個字、音量差也不明顯的插段，判回給前後那個人。
    for i in range(1, len(words) - 1):
        j = i
        while j < len(words) and j - i < 3 and words[j]["speaker"] != words[i - 1]["speaker"]:
            j += 1
        if (j - i <= 2 and j < len(words) and words[j]["speaker"] == words[i - 1]["speaker"]
                and all(w["margin"] < 10 for w in words[i:j])):
            for w in words[i:j]:
                w["speaker"] = words[i - 1]["speaker"]

    # 同一人連續說的字併成一段；換人或停頓超過 1 秒就斷開
    segments = []
    for w in words:
        if segments and segments[-1]["speaker"] == w["speaker"] and w["start"] - segments[-1]["end"] < 1.0:
            segments[-1]["words"].append(w)
            segments[-1]["end"] = w["end"]
        else:
            segments.append({"start": w["start"], "end": w["end"], "speaker": w["speaker"], "words": [w]})
    for s in segments:
        s["text"] = "".join(w["word"] for w in s["words"])

    fillers = []
    g, r = int(GAP / hop), int(SEARCH / hop)
    for s in segments:
        ws = s["words"]
        for i, w in enumerate(ws):
            if w["word"].strip("，。,.?？!！ ") not in FILLERS:
                continue
            a, b = int(w["start"] / hop), int(w["end"] / hop)
            q = quiet[w["speaker"]]
            if has_gap(q, a - r, a + g, g) and has_gap(q, b - g, b + r, g):
                fillers.append({"speaker": w["speaker"], "word": w["word"], "start": w["start"], "end": w["end"],
                                "context": "".join(x["word"] for x in ws[max(i - 6, 0):i + 7])})

    lines = [f"# 逐字稿：{cfg['title']}", "",
             "mlx-whisper（whisper-large-v3-turbo）轉錄各軌混音，再依各軌音量逐字判定說話者。未校對；換人處的一兩個字常被判給對方。時間是原始錄音的時間碼。", ""]
    lines += [f"[{hms(s['start'])}] {s['speaker']}：{s['text']}\n" for s in segments]
    (ep / "work/逐字稿.md").write_text("\n".join(lines))
    (ep / "work/segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=0))
    (ep / "work/fillers.json").write_text(json.dumps(fillers, ensure_ascii=False, indent=1))
    low = sum(1 for w in words if w["margin"] < 3)
    print(f"字 {len(words)}，段落 {len(segments)}，說話者判定不確定（差 <3 dB）{low} 字，可刪填充詞 {len(fillers)}")


if __name__ == "__main__":
    main(sys.argv[1])
