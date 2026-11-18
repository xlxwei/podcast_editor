"""新的一集從錄音到可以開始剪：檢查各軌格式、混音、轉錄、判定說話者、建立初始的 edl。

用法：python tools/prepare.py <集資料夾>
需要：<集資料夾>/episode.json（沒有的話會依資料夾裡的音檔產生一份草稿，填好再跑一次）
已經做過的步驟會跳過；要重做就刪掉對應的檔案（寫在每一步的訊息裡）。
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".aif", ".aiff")
MODEL = "mlx-community/whisper-large-v3-turbo"


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=codec_name,sample_rate,channels", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out


def main(ep):
    ep = Path(ep)
    cfg_path = ep / "episode.json"
    if not cfg_path.exists():
        files = sorted(f.name for f in ep.iterdir() if f.suffix.lower() in AUDIO)
        draft = {"title": ep.name, "language": "zh", "limit_min": 40,
                 "tracks": [{"key": f"t{i + 1}", "file": f, "speaker": f"說話者{i + 1}", "gain_db": 0} for i, f in enumerate(files)]}
        cfg_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
        sys.exit(f"已產生 {cfg_path}。請確認 tracks 只列每人一軌的錄音，把 speaker 改成名字，再跑一次。")
    cfg = config.load(ep)
    work = ep / "work"
    (work / "transcribe").mkdir(parents=True, exist_ok=True)

    # 1. 各軌統一成 48 kHz、16-bit、mono 的 wav（頁面和輸出都假設這個格式）
    changed = False
    for t in cfg["tracks"]:
        src = ep / t["file"]
        if not src.exists():
            sys.exit(f"找不到 {src}")
        if probe(src) != "pcm_s16le,48000,1":
            dst = work / "tracks" / f"{t['key']}.wav"
            dst.parent.mkdir(exist_ok=True)
            print(f"[1] {t['file']} 轉成 48 kHz 16-bit mono → {dst.relative_to(ep)}")
            run("ffmpeg", "-nostats", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", dst)
            t["source"], t["file"] = t["file"], str(dst.relative_to(ep))
            changed = True
    if changed:
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

    # 2. 混音：給轉錄用的 16 kHz wav，和頁面上「原始」按鈕播的 mp3
    mix16, mp3 = work / "dialogue_mix16k.wav", work / "原始_mix.mp3"
    if not (mix16.exists() and mp3.exists()):
        print("[2] 混音")
        n = len(cfg["tracks"])
        ins = [x for t in cfg["tracks"] for x in ("-i", ep / t["file"])]
        graph = ";".join(f"[{i}]volume={t['gain_db']}dB[a{i}]" for i, t in enumerate(cfg["tracks"]))
        graph += ";" + "".join(f"[a{i}]" for i in range(n)) + f"amix=inputs={n}:normalize=0[m]"
        run("ffmpeg", "-nostats", "-loglevel", "error", "-y", *ins, "-filter_complex", graph, "-map", "[m]", "-ar", "16000", mix16)
        run("ffmpeg", "-nostats", "-loglevel", "error", "-y", "-i", mix16, "-af", "loudnorm=I=-16:TP=-1.5", "-b:a", "96k", mp3)
    else:
        print("[2] 混音已經有了（要重做就刪 work/dialogue_mix16k.wav 和 work/原始_mix.mp3）")

    # 3. 轉錄（一小時的錄音在 Apple Silicon 上大約幾分鐘；第一次會下載模型，約 1.6 GB）
    tj = work / "transcribe/dialogue_mix16k.json"
    if not tj.exists():
        print("[3] 轉錄，請稍等")
        import mlx_whisper
        res = mlx_whisper.transcribe(str(mix16), path_or_hf_repo=MODEL, language=cfg.get("language", "zh"), word_timestamps=True)
        tj.write_text(json.dumps(res, ensure_ascii=False))
    else:
        print("[3] 轉錄已經有了（要重做就刪 work/transcribe/dialogue_mix16k.json）")

    # 4. 音量包絡、長停頓、逐字判定說話者
    print("[4] 包絡與說話者判定")
    import envelope
    import merge_transcript
    envelope.main(ep)
    merge_transcript.main(ep)

    # 5. 章節：先放一個整集的段落，之後由 Claude 讀逐字稿分章節、寫摘要
    if not (ep / "sections.json").exists():
        (ep / "sections.json").write_text(json.dumps(
            [{"chapter": "全集（還沒分章節）", "sections": [{"id": "1-1", "start": "0:00", "title": "全集", "summary": "還沒分段。"}]}],
            ensure_ascii=False, indent=1))
        print("[5] 已建立只有一段的 sections.json")

    # 6. 初始 edl：整集一個區塊，前後有停頓的填充詞先標成建議刪
    if not (ep / "edl.json").exists():
        import numpy as np
        env = np.load(work / "envelope.npz")
        dur = round(len(env[cfg["tracks"][0]["key"]]) * float(env["hop"]), 2)
        fillers = json.loads((work / "fillers.json").read_text())
        removes = [{"from": f["start"], "to": f["end"], "note": f"填充詞「{f['word']}」", "by": "claude", "kind": "tech"}
                   for f in fillers if 0 < f["start"] and f["end"] < dur]
        (ep / "edl.json").write_text(json.dumps(
            {"note": "from / to 是原始錄音的秒數，to 不含。blocks 是照播出順序排的保留區塊，只有搬移順序時才需要多於一塊；"
                     "所有刪減都寫在 remove。格式說明見工具 repository 的 CLAUDE.md。",
             "blocks": [{"from": 0, "to": dur, "keep": "全集", "cut_after": ""}], "remove": removes, "keep": []},
            ensure_ascii=False, indent=1))
        print(f"[6] 已建立 edl.json，先標了 {len(removes)} 個填充詞")

    # 7. 這一集自己的版本紀錄（頁面上每個動作會自動 commit）
    inside = subprocess.run(["git", "-C", str(ep), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    ignored = subprocess.run(["git", "-C", str(ep), "check-ignore", "-q", "."], capture_output=True).returncode == 0
    if inside.returncode != 0 or ignored:
        (ep / ".gitignore").write_text("*.wav\n*.mp3\n*.m4a\n*.flac\n*.aif\n*.aiff\n*.npz\nwork/transcribe/\nwork/tracks/\nwork/pieces.json\nwork/scan_items.json\n*.html\n.DS_Store\n")
        run("git", "-C", ep, "init", "-q")
        run("git", "-C", ep, "add", "-A")
        run("git", "-C", ep, "commit", "-q", "-m", "準備完成：轉錄、章節草稿、初始 edl")
        print("[7] 已在這一集的資料夾建立 git 版本紀錄（不含音檔）")

    print(f"\n準備好了。接下來：\n  python tools/serve.py {ep}\n然後用瀏覽器開 http://localhost:8765/review")


if __name__ == "__main__":
    main(sys.argv[1])
