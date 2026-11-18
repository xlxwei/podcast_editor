"""對話軌的能量包絡與停頓偵測。

用法：python tools/envelope.py <集資料夾>
需要：episode.json 和它列出的各軌 wav
輸出：<集資料夾>/work/envelope.npz、work/pauses.json
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

HOP = 0.01          # 包絡解析度 10 ms
PAUSE_MIN = 1.2     # 所有人同時沉默超過這個長度才縮
PAUSE_KEEP = 0.6    # 縮成這個長度


def envelope_db(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    n = int(sr * HOP)
    x = x[: len(x) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((x ** 2).mean(1)) + 1e-9)


def speech_mask(db, other_db):
    floor = np.percentile(db, 5)
    # 比自己的底噪高 15 dB，且比其他軌大（排除串音）
    active = (db > floor + 15) & (db > other_db - 3)
    # 平滑：100 ms 內的空隙補起來
    k = int(0.1 / HOP)
    smoothed = np.convolve(active.astype(float), np.ones(k), "same") > 0
    return smoothed


def runs(mask, value):
    """回傳 mask == value 的連續區段 [(start_s, end_s)]。"""
    padded = np.concatenate([[not value], mask == value, [not value]]).astype(int)
    d = np.diff(padded)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s * HOP, e * HOP) for s, e in zip(starts, ends)]


def main(ep):
    ep = Path(ep)
    tracks = config.load(ep)["tracks"]
    db = [envelope_db(ep / t["file"]) for t in tracks]
    n = min(len(d) for d in db)
    db = [d[:n] for d in db]
    on = [speech_mask(d, np.max([o for j, o in enumerate(db) if j != i], axis=0)) for i, d in enumerate(db)]
    silence = ~np.any(on, axis=0)
    pauses = [(s, e) for s, e in runs(silence, True) if e - s > PAUSE_MIN]
    (ep / "work").mkdir(exist_ok=True)
    arrays = {t["key"]: d for t, d in zip(tracks, db)} | {t["key"] + "_on": m for t, m in zip(tracks, on)}
    np.savez_compressed(ep / "work/envelope.npz", hop=HOP, **arrays)
    saved = sum(e - s - PAUSE_KEEP for s, e in pauses)
    (ep / "work/pauses.json").write_text(json.dumps(
        {"min": PAUSE_MIN, "keep": PAUSE_KEEP, "saved_s": round(saved, 1),
         "pauses": [[round(s, 2), round(e, 2)] for s, e in pauses]}, indent=1))
    print(f"長停頓 {len(pauses)} 處，縮短後省 {saved:.1f} 秒；" + "，".join(f"{t['speaker']}說話 {m.mean():.0%}" for t, m in zip(tracks, on)))


if __name__ == "__main__":
    main(sys.argv[1])
