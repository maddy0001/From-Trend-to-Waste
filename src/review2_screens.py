"""Screenshots of the Review 2 dashboard for the 8-slide deck -> outputs/figures/review2/

Reuses the headless-Chrome helpers of src/dashboard_screenshots.py. Saves two crops:
  r2_overview.png  title, progress, key numbers and insights (Overview tab)
  r2_q1.png        Q1 with its filter, chart and finding (Findings tab) - shows the interactive layout
"""
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import websockets
from PIL import Image

from . import dashboard_screenshots as ds
from .config import FIG, ROOT

OUT = FIG / "review2"
ds.READY_JS = """(() => document.querySelectorAll('[role=tab]').length >= 4
  && !document.querySelector('[data-testid="stStatusWidget"]'))()"""


async def capture(ws_url, url):
    shots = {}
    async with websockets.connect(ws_url, max_size=None) as ws:
        cdp = ds.CDP(ws)
        await cdp.call("Page.enable")
        for idx, name in ((0, "overview"), (2, "findings")):
            await cdp.viewport(1000)
            await cdp.call("Page.navigate", url=url)
            await asyncio.sleep(2)
            await cdp.wait_ready()
            await cdp.js(f"document.querySelectorAll('[role=tab]')[{idx}].click()")
            await cdp.viewport(9000)
            await asyncio.sleep(5)  # charts render after the script finishes
            lay = await cdp.js(ds.LAYOUT_JS)
            await cdp.viewport(lay["block"]["y"] + lay["block"]["h"] + 40)
            await asyncio.sleep(3)
            lay = await cdp.js(ds.LAYOUT_JS)
            shot = await cdp.call("Page.captureScreenshot", format="png")
            path = OUT / f"full_{name}.png"
            path.write_bytes(__import__("base64").b64decode(shot["data"]))
            shots[name] = (path, lay)
    return shots


def crop(path, lay, y0, y1, out):
    k = ds.SCALE
    Image.open(path).crop((int(56 * k), int(y0 * k), int(1384 * k), int(y1 * k))).save(OUT / out)
    (OUT / out.replace(".png", ".json")).write_text(json.dumps({"aspect": 1328 / (y1 - y0)}))
    print("saved", OUT / out)


def main():
    if not ds.CHROME.exists():
        print("Google Chrome not found: skip Review 2 screenshots")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    port, dbg = ds.free_port(), ds.free_port()
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "review2_app.py"),
                            "--server.headless", "true", "--server.port", str(port)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp = Path(tempfile.mkdtemp())
    chrome = None
    try:
        ds.wait_http(f"http://127.0.0.1:{port}/_stcore/health")
        chrome = subprocess.Popen([str(ds.CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                                   f"--remote-debugging-port={dbg}", f"--user-data-dir={tmp}", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ds.wait_http(f"http://127.0.0.1:{dbg}/json/version")
        pages = json.loads(ds.wait_http(f"http://127.0.0.1:{dbg}/json/list"))
        ws_url = next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page")
        shots = asyncio.run(capture(ws_url, f"http://127.0.0.1:{port}/"))
    finally:
        for proc in (chrome, app):
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
        shutil.rmtree(tmp, ignore_errors=True)
    path, lay = shots["overview"]
    els = lay["elements"]
    table = [e for e in els if e["testid"] == "stDataFrame"][0]
    crop(path, lay, lay["title"]["y"] - 20, table["y"] + table["h"] + 12, "r2_overview.png")
    path, lay = shots["findings"]
    heads = [e for e in lay["elements"] if e["testid"] == "stHeading"]
    q1 = next(e for e in heads if e["text"].startswith("Q1"))
    q2 = next(e for e in heads if e["text"].startswith("Q2"))
    crop(path, lay, lay["tablist"]["y"] - 10, q2["y"] - 30, "r2_q1.png")


if __name__ == "__main__":
    main()
