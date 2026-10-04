"""plotly 그림 → PNG. 소유자: E. (kaleido 가 이 환경에서 멈춰서 만든 대체 경로)

    python scripts/render_plotly_png.py fig.html out.png --width 1600 --height 1200
    python scripts/render_plotly_png.py a.html b.html --out-dir outputs/presentation

파이썬에서 쓸 때는 `render_figure(fig, path)` (Figure 를 임시 HTML 로 적고 찍는다).

**왜 이게 필요한가**

1. `fig.write_image(...)` (kaleido) 가 이 작업 환경에서 **멈춘다**. 단순 선 그래프 하나도 3분 타임아웃에
   걸렸다 (plotly 7.1 + kaleido v1 이 Chrome 을 띄우고 기다리는 구간). 다른 트랙에서도 발표용 그림을
   PNG 로 뽑으려 하면 같은 곳에서 막힌다.
2. 헤드리스 브라우저로 직접 찍으면 **3D(WebGL) 가 빈 화면으로 나온다**. `Page.captureScreenshot` 이
   WebGL 캔버스를 합성하지 못한다. 그래서 페이지 안에서 **`Plotly.toImage`** 를 불러 plotly 가 직접
   합성한 PNG 를 받아온다. 이것이 이 스크립트의 핵심이다.
3. 소프트웨어 WebGL 플래그(`--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`)가 없으면
   헤드리스에서 WebGL 자체가 안 뜬다.

`websockets` 와 Microsoft Edge 가 필요하다. Edge 경로는 `--edge` 로 바꿀 수 있다.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
#: 헤드리스에서 WebGL(plotly 3D)을 소프트웨어로 그리게 하는 플래그
GL_FLAGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist"]
#: 그림이 그려졌는지 보는 선택자
READY_JS = ("document.querySelectorAll('.js-plotly-plot canvas,"
            " .js-plotly-plot .main-svg').length")


class _Page:
    """CDP 한 탭. 필요한 것만 감싼다."""

    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method: str, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr: str, await_promise: bool = False):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True,
                            awaitPromise=await_promise)
        return r.get("result", {}).get("value")


def _launch(edge: str, port: int, profile: Path) -> subprocess.Popen:
    return subprocess.Popen([edge, "--headless=new", f"--remote-debugging-port={port}",
                             f"--user-data-dir={profile}", "--no-first-run", "--hide-scrollbars",
                             "--allow-file-access-from-files", *GL_FLAGS, "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ws_url(port: int, timeout: float = 60.0) -> str:
    end = time.time() + timeout
    while time.time() < end:
        try:
            tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
            pages = [t for t in tabs if t["type"] == "page"]
            if pages:
                return pages[0]["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"헤드리스 브라우저 CDP({port})에 붙지 못했다")


async def _shoot(jobs: list[tuple[Path, Path, int, int]], *, edge: str, port: int,
                 settle_s: float) -> list[Path]:
    import websockets                                   # 지연 import (테스트·import 를 가볍게)

    profile = Path(tempfile.gettempdir()) / f"airis_render_{uuid.uuid4().hex[:8]}"
    proc = _launch(edge, port, profile)
    done: list[Path] = []
    try:
        async with websockets.connect(_ws_url(port), max_size=400 * 1024 * 1024) as ws:
            page = _Page(ws)
            await page.call("Page.enable")
            await page.call("Runtime.enable")
            for html, out, w, h in jobs:
                await page.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                                deviceScaleFactor=1, mobile=False)
                url = "file:///" + urllib.parse.quote(str(html.resolve()).replace("\\", "/"))
                await page.call("Page.navigate", url=url)
                for _ in range(150):
                    await asyncio.sleep(0.4)
                    if await page.js(READY_JS):
                        break
                await asyncio.sleep(settle_s)
                # 카메라 버튼 같은 레이아웃 요소는 figure 에서 지운다 (toImage 가 다시 그린다)
                await page.call("Runtime.evaluate", awaitPromise=True, returnByValue=True,
                                expression="Plotly.relayout(document.querySelector"
                                           "('.js-plotly-plot'), {updatemenus: [], sliders: []})"
                                           ".then(() => true)")
                # **핵심**: 화면 캡처가 아니라 plotly 가 합성한 PNG 를 받는다 (WebGL 이 빈 채로 찍히는 것 회피)
                data = await page.js(
                    f"Plotly.toImage(document.querySelector('.js-plotly-plot'),"
                    f" {{format:'png', width:{w}, height:{h}}})", await_promise=True) or ""
                if not data.startswith("data:image/png;base64,"):
                    raise RuntimeError(f"{html.name}: plotly 가 그림을 내주지 않았다 ({data[:60]})")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(base64.b64decode(data.split(",", 1)[1]))
                done.append(out)
    finally:
        proc.terminate()
    return done


def render_html(html_paths, out_paths=None, *, width: int = 1600, height: int = 1200,
                edge: str = DEFAULT_EDGE, port: int = 9400, settle_s: float = 1.2) -> list[Path]:
    """plotly 그림이 담긴 HTML 들을 PNG 로 찍는다. 브라우저는 한 번만 띄운다."""
    htmls = [Path(p) for p in html_paths]
    outs = ([Path(p) for p in out_paths] if out_paths
            else [h.with_suffix(".png") for h in htmls])
    if len(outs) != len(htmls):
        raise ValueError("html 과 out 개수가 다르다")
    return asyncio.run(_shoot([(h, o, width, height) for h, o in zip(htmls, outs)],
                              edge=edge, port=port, settle_s=settle_s))


def render_figure(fig, out_path, *, width: int = 1600, height: int = 1200, **kw) -> Path:
    """plotly Figure 하나를 PNG 로. `fig.write_image` 자리에 쓴다."""
    tmp = Path(tempfile.gettempdir()) / f"airis_fig_{uuid.uuid4().hex[:8]}.html"
    try:
        fig.write_html(tmp, include_plotlyjs=True, full_html=True,
                       config=dict(staticPlot=True, displayModeBar=False))
        return render_html([tmp], [out_path], width=width, height=height, **kw)[0]
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="plotly HTML → PNG (kaleido 대신 헤드리스 브라우저)")
    ap.add_argument("paths", nargs="+", help="HTML 들, 또는 'in.html out.png' 두 개")
    ap.add_argument("--out-dir", default=None, help="PNG 를 모아 둘 폴더 (기본: HTML 옆)")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--edge", default=DEFAULT_EDGE)
    ap.add_argument("--port", type=int, default=9400)
    a = ap.parse_args(argv)

    if len(a.paths) == 2 and a.paths[1].lower().endswith(".png"):
        htmls, outs = [Path(a.paths[0])], [Path(a.paths[1])]
    else:
        htmls = [Path(p) for p in a.paths]
        outs = ([Path(a.out_dir) / (h.stem + ".png") for h in htmls] if a.out_dir else None)
    for p in render_html(htmls, outs, width=a.width, height=a.height, edge=a.edge, port=a.port):
        print(f"{p}  {p.stat().st_size / 1e3:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
