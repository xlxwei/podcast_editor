"""剪輯頁面用的本機伺服器，也提供靜態檔案，支援 Range 請求，讓瀏覽器可以跳到音檔中間播放。

用法：python tools/serve.py <集資料夾> [port]
逐字稿頁：http://localhost:<port>/review（tools/review.html，邏輯在 review_api.py）。
刀點微調頁：http://localhost:<port>/tune（tools/tune.html）。
"""
import json
import os
import re
import struct
import subprocess
import sys
import threading
import wave
from datetime import date
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import render_preview as rp  # noqa: E402
import review_api  # noqa: E402

SAVE_LOCK = threading.Lock()


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "remaining", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(65536, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


class TuneHandler(RangeHandler):
    """刀點微調頁（/tune）用的 API：列出每一刀、切一小段原始兩軌、把拉好的位置存回 edl.json、重新輸出。"""
    ep_dir = None  # 由 __main__ 設定
    _review = None

    @classmethod
    def review(cls):
        if cls._review is None:
            cls._review = review_api.Review(cls.ep_dir)
        return cls._review

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _edl(self):
        return json.loads((self.ep_dir / "edl.json").read_text())

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/tune", "/review"):
                body = (TOOLS / (url.path[1:] + ".html")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/review":
                self._json(self.review().static())
            elif url.path == "/api/preview":
                body = self.review().preview([tuple(float(x) for x in p.split("-")) for p in q["p"].split(",")])
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/cuts":
                self._json(self.cuts())
            elif url.path == "/api/words":
                a, b = float(q["start"]), float(q["end"])
                segs = json.loads((self.ep_dir / "work/segments.json").read_text())
                self._json([{"t": w["start"], "e": w["end"], "w": w["word"], "who": g["speaker"]}
                            for g in segs if g["end"] >= a and g["start"] <= b
                            for w in g["words"] if a <= w["start"] <= b])
            elif url.path == "/api/clip":
                self.clip(float(q["start"]), float(q["end"]))
            else:
                super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 瀏覽器停止播放時會中斷連線，不是錯誤
        except Exception as e:  # 讓頁面看得到錯誤訊息
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def cuts(self):
        ep = rp.Episode(self.ep_dir)
        edl = self._edl()
        blocks, pieces, fades = rp.plan(ep, edl)
        ends, t = [], 0.0  # 每個片段的結尾在試聽版的位置
        for i, (a, b) in enumerate(pieces):
            t += (b - a) - (fades[i] if i else 0)
            ends.append(t)

        def join(a):
            """原始時間 a 之前最後一個片段：(試聽版位置, 那個片段的起點)。"""
            i = max((i for i, p in enumerate(pieces) if p[1] <= a + 1e-3), key=lambda i: pieces[i][1], default=0)
            return ends[i], pieces[i][0]

        def after(b):
            return min((p[1] for p in pieces if abs(p[0] - b) < 1e-3), default=b + 10)

        out = []
        for i, r in enumerate(edl["remove"]):
            if any(blk["a"] < r["from"] and r["to"] < blk["b"] for blk in blocks):
                o, lo = join(r["from"])
                out.append(dict(id=f"r{i}", kind="remove", a=r["from"], b=r["to"], fade=r.get("fade", rp.XFADE),
                                note=r.get("note", ""), out=o, lo=lo, hi=after(r["to"])))
        for i in range(1, len(blocks)):
            p, n = blocks[i - 1], blocks[i]
            o, lo = join(p["b"])
            out.append(dict(id=f"b{i}", kind="block", a=p["b"], b=n["a"], fade=n.get("fade", rp.XFADE),
                            note=f"區塊接點 {rp.mmss(p['b'])} ‖ {rp.mmss(n['a'])}：…{ep.context(p['b'], True)} ‖ {ep.context(n['a'], False)}…",
                            out=o, lo=lo, hi=after(n["a"])))
        out.sort(key=lambda c: c["out"])
        tracks = rp.config.load(self.ep_dir)["tracks"]
        return dict(cuts=out, default_fade=rp.XFADE, episode=self.ep_dir.name,
                    tracks=[dict(speaker=t["speaker"], gain=10 ** (t["gain_db"] / 20)) for t in tracks])

    def clip(self, start, end):
        """原始各軌的一小段，多聲道 WAV：第 k 聲道是 episode.json 的第 k 軌。"""
        if not 0 < end - start <= 60:
            raise ValueError("片段長度要在 60 秒以內")
        chans = []
        for name in [t["file"] for t in rp.config.load(self.ep_dir)["tracks"]]:
            with wave.open(str(self.ep_dir / name)) as w:
                sr = w.getframerate()
                a = max(0, int(start * sr))
                w.setpos(min(a, w.getnframes()))
                chans.append(w.readframes(int((end - start) * sr)))
        n, k = min(len(c) for c in chans) // 2, len(chans)
        data = bytearray(n * 2 * k)
        for j, c in enumerate(chans):   # 交錯排成多聲道
            data[2 * j::2 * k], data[2 * j + 1::2 * k] = c[0:n * 2:2], c[1:n * 2:2]
        head = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, k, sr, sr * 2 * k, 2 * k, 16) \
            + b"data" + struct.pack("<I", len(data))
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(head) + len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(head + bytes(data))

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1"):
                return self._json({"error": "只接受本機頁面的請求"}, 403)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            if self.path == "/api/save":
                self._json(self.save(body))
            elif self.path in ("/api/mark", "/api/keep"):
                with SAVE_LOCK:
                    rv = self.review()
                    msg, state = rv.mark(body["first"], body["last"]) if self.path == "/api/mark" else \
                        rv.keep(body.get("first"), body.get("last"), body.get("remove"))
                    self.commit("逐字稿頁：" + msg)
                self._json({"ok": True, "state": state})
            elif self.path == "/api/render":
                py = sys.executable
                log = ""
                for tool in ("render_preview.py",):
                    r = subprocess.run([py, str(TOOLS / tool), str(self.ep_dir)], capture_output=True, text=True, cwd=TOOLS.parent)
                    log += r.stdout[-400:] + r.stderr[-800:]
                    if r.returncode:
                        return self._json({"error": log}, 500)
                self.commit("微調頁重新輸出試聽版")
                self._json({"ok": True, "log": log.strip().splitlines()[-3:]})
            else:
                self._json({"error": "沒有這個路徑"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def commit(self, msg):
        """把這一集資料夾的變動記進 git。不是 git repository 或沒有變動就什麼都不做，失敗也不影響儲存。"""
        try:
            git = ["git", "-C", str(self.ep_dir)]
            subprocess.run(git + ["add", "--", "."], capture_output=True, timeout=20)
            subprocess.run(git + ["commit", "-q", "-m", f"{self.ep_dir.name} {msg}", "--", "."], capture_output=True, timeout=20)
        except Exception:
            pass

    def save(self, body):
        a, b, fade = round(float(body["a"]), 3), round(float(body["b"]), 3), round(float(body["fade"]), 3)
        if not 0.002 <= fade <= 0.5:
            raise ValueError("淡化要在 2 到 500 毫秒之間")
        path = self.ep_dir / "edl.json"
        with SAVE_LOCK:
            edl = json.loads(path.read_text())
            backup = self.ep_dir / f"work/edl_微調前_{date.today():%m%d}.json"
            if not backup.exists():
                backup.write_text(path.read_text())
            kind, i = body["id"][0], int(body["id"][1:])
            if kind == "r":
                if a >= b:
                    raise ValueError("剪掉範圍的起點要在終點之前")
                item = edl["remove"][i]
                item["from"], item["to"] = a, b
            elif kind == "b":
                edl["blocks"][i - 1]["to"] = a
                item = edl["blocks"][i]
                item["from"] = b
            else:
                raise ValueError("不認得的編號")
            if abs(fade - rp.XFADE) < 1e-6:
                item.pop("fade", None)
            else:
                item["fade"] = fade
            rp.plan(rp.Episode(self.ep_dir), edl)  # 存之前先確認整份 edl 還排得出來
            path.write_text(json.dumps(edl, ensure_ascii=False, indent=1))
            self.commit(f"微調頁：{body['id']} 改成 {a}–{b}，淡化 {fade * 1000:.0f} ms")
        return {"ok": True, "a": a, "b": b, "fade": fade}


if __name__ == "__main__":
    root = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    TuneHandler.ep_dir = Path(root).resolve()
    ThreadingHTTPServer(("127.0.0.1", port), partial(TuneHandler, directory=root)).serve_forever()
