"""每一集的設定：<集資料夾>/episode.json。

title      顯示在頁面和清單上的名稱
limit_min  成品對話長度上限（分鐘）
language   轉錄的語言，預設 zh（中文會轉成台灣繁體）
tracks     每人一軌，同長、時間對齊。key 是 envelope.npz 裡的名稱，file 是 wav 檔名，
           speaker 是逐字稿上的名字，gain_db 是混音和比較音量前先補的增益（mic 比較小聲的人填正數）
"""
import json
from pathlib import Path


def load(ep):
    cfg = json.loads((Path(ep) / "episode.json").read_text())
    if len(cfg["tracks"]) < 2:
        raise ValueError("episode.json 的 tracks 至少要兩軌")
    return cfg
