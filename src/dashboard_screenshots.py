"""Screenshots of every dashboard tab -> outputs/figures/dashboard/ (used by the deck's dashboard slides).

Starts the Streamlit app and a headless Chrome, clicks through the tabs over the Chrome DevTools Protocol,
and saves for each tab a full-length PNG plus the positions of its widgets (CSS px) in a JSON file, so the deck
can crop to the important part and number the controls. Skips quietly when Chrome is not installed.
"""
import asyncio
import base64
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

from .config import FIG, ROOT

OUT = FIG / "dashboard"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
WIDTH, SCALE = 1440, 2
# Example for the pre-launch tab: a yellow scuba-crepe pair of culottes at the top of its price range, bought deep
RISKY = {"category": "culottes", "color": "yellow", "fabric": "scuba crepe", "season_type": "SS", "price_tier": "0.9",
         "release_month": "3", "n_shops": "14", "initial_per_shop": "12"}
SHOTS = [(0, "tab0_overview", {}), (1, "tab_data_eda", {}), (2, "tab1_prelaunch_risk", RISKY),
         (3, "tab2_markdown_watchlist", {}), (4, "tab3_customers_survey", {}), (5, "tab4_impact_simulator", {}),
         (6, "tab5_financials", {})]

READY_JS = """(() => {
  const tabs = document.querySelectorAll('[role=tab]');
  const running = document.querySelector('[data-testid="stStatusWidget"]');
  const imgs = [...document.images].every(i => i.complete && i.naturalWidth > 0);
  return tabs.length >= 7 && !running && imgs;
})()"""

LAYOUT_JS = """(() => {
  const panel = [...document.querySelectorAll('[role=tabpanel]')].find(p => p.offsetParent !== null && !p.hidden);
  const block = document.querySelector('[data-testid="stMainBlockContainer"]');
  const box = e => { const r = e.getBoundingClientRect(); return {x: r.left, y: r.top, w: r.width, h: r.height}; };
  const want = ['stMetric', 'stSelectbox', 'stSlider', 'stCheckbox', 'stDataFrame', 'stImage', 'stHeading',
                'stMarkdown', 'stRadio', 'stMultiSelect', 'stCaption', 'stSelectSlider', 'stAlert', 'stVegaLiteChart', 'stArrowVegaLiteChart'];
  const els = panel ? [...panel.querySelectorAll('[data-testid]')].filter(e => want.includes(e.dataset.testid)) : [];
  return {
    block: box(block), panel: panel ? box(panel) : null,
    tablist: box(document.querySelector('[role=tablist]')),
    title: box(document.querySelector('h1')),
    elements: els.map(e => ({testid: e.dataset.testid, text: (e.innerText || '').trim().slice(0, 160), ...box(e)}))
             .filter(e => e.w > 0 && e.h > 0),
  };
})()"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.read()
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(url)


class CDP:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r["result"].get("value")

    async def wait_ready(self, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if await self.js(READY_JS):
                await asyncio.sleep(1.5)
                if await self.js(READY_JS):
                    return
            await asyncio.sleep(0.5)
        raise TimeoutError("dashboard did not finish rendering")

    async def viewport(self, height):
        await self.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=int(height), deviceScaleFactor=SCALE, mobile=False)


async def capture(ws_url, app_url):
    async with websockets.connect(ws_url, max_size=None) as ws:
        cdp = CDP(ws)
        await cdp.call("Page.enable")
        await cdp.call("Runtime.enable")
        for idx, name, params in SHOTS:
            await cdp.viewport(1000)
            q = "&".join(f"{k}={urllib.request.quote(v)}" for k, v in params.items())
            await cdp.call("Page.navigate", url=app_url + ("?" + q if q else ""))
            await asyncio.sleep(2)
            await cdp.wait_ready()
            await cdp.js(f"document.querySelectorAll('[role=tab]')[{idx}].click()")
            await asyncio.sleep(2)
            await cdp.wait_ready()
            await cdp.viewport(9000)  # tall viewport so nothing sits inside a scrolled container
            await asyncio.sleep(2)
            lay = await cdp.js(LAYOUT_JS)
            height = max(lay["block"]["y"] + lay["block"]["h"], lay["panel"]["y"] + lay["panel"]["h"]) + 40
            await cdp.viewport(height)
            await asyncio.sleep(2)
            lay = await cdp.js(LAYOUT_JS)
            shot = await cdp.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
            (OUT / f"{name}.png").write_bytes(base64.b64decode(shot["data"]))
            lay.update(scale=SCALE, width=WIDTH, height=height, params=params)
            (OUT / f"{name}.json").write_text(json.dumps(lay, indent=1))
            print(f"saved {OUT / name}.png ({len(lay['elements'])} widgets)")


# Crops for the deck: (output, source tab, top, bottom, numbered markers). A selector is (testid, text it contains, nth)
# (captions count as stMarkdown);
# "tablist" and "title" refer to the tab bar and the page title.
CROPS = [
    ("crop_overview", "tab0_overview", ("title", -24), ("stImage", "", 0, "bottom", 14),
     [("tablist",), ("stMetric", "", 0), ("stMarkdown", "Key insights", 0), ("stImage", "", 0)]),
    ("crop_eda_log", "tab_data_eda", ("tablist", -12), ("stDataFrame", "", 1, "bottom", 12),
     [("stDataFrame", "", 0), ("stMultiSelect", "", 0), ("stDataFrame", "", 1)]),
    ("crop_eda", "tab_data_eda", ("stMarkdown", "3 · Explore", 0, "top", -8), ("stVegaLiteChart", "", 1, "bottom", 12),
     [("stSelectbox", "", 0), ("stSelectbox", "Filter by", 0), ("stMetric", "", 0), ("stVegaLiteChart", "", 0)]),
    ("crop_prelaunch", "tab1_prelaunch_risk", ("tablist", -12), ("stMarkdown", "Suggested action", 0, "bottom", 14),
     [("stSelectbox", "", 0), ("stSlider", "", 0), ("stMetric", "", 0), ("stMarkdown", "Suggested action", 0)]),
    ("crop_watchlist", "tab2_markdown_watchlist", ("tablist", -12), ("stMarkdown", "Action for each flagged", 0, "bottom", 12),
     [("stSlider", "", 0), ("stMetric", "", 0), ("stDataFrame", "", 0)]),
    ("crop_customers", "tab3_customers_survey", ("tablist", -12), ("stDataFrame", "", 0, "bottom", 12),
     [("stMetric", "", 0), ("stDataFrame", "", 0)]),
    ("crop_survey", "tab3_customers_survey", ("stHeading", "Consumer survey", 0, "top", -8), ("stMetric", "", 7, "bottom", 12),
     [("stMetric", "", 4)]),
    ("crop_simulator", "tab4_impact_simulator", ("tablist", -12), ("stMarkdown", "Demand = observed", 0, "bottom", 12),
     [("stSelectbox", "", 0), ("stMetric", "", 0), ("stMarkdown", "tonnes", 0)]),
    ("crop_financials", "tab5_financials", ("tablist", -12), ("stDataFrame", "", 0, "bottom", 12),
     [("stSlider", "Assumed unit cost", 0), ("stMetric", "", 0), ("stDataFrame", "", 0)]),
]
X0, X1 = 56, 1384  # horizontal crop (CSS px): content column plus a small margin


def find(lay, sel):
    if sel[0] in ("tablist", "title"):
        return lay[sel[0]]
    hits = [e for e in lay["elements"] if e["testid"] == sel[0] and sel[1] in e["text"]]
    if len(hits) <= sel[2]:
        raise KeyError(f"{sel} not found")
    return hits[sel[2]]


def edge(lay, spec):
    if spec[0] in ("tablist", "title"):
        return lay[spec[0]]["y"] + spec[1]
    e = find(lay, spec[:3])
    return (e["y"] + e["h"] if spec[3] == "bottom" else e["y"]) + spec[4]


def make_crops():
    """Cut the deck crops from the saved screenshots and record where the numbered markers go (as fractions)."""
    from PIL import Image
    for out, src, top, bottom, marks in CROPS:
        lay = json.loads((OUT / f"{src}.json").read_text())
        k = lay["scale"]
        y0, y1 = edge(lay, top), edge(lay, bottom)
        Image.open(OUT / f"{src}.png").crop((int(X0 * k), int(y0 * k), int(X1 * k), int(y1 * k))).save(OUT / f"{out}.png")
        markers = []
        for sel in marks:
            e = find(lay, sel)
            markers.append({"fx": (e["x"] - X0) / (X1 - X0), "fy": (e["y"] - y0) / (y1 - y0),
                            "fw": e["w"] / (X1 - X0), "fh": e["h"] / (y1 - y0), "text": e.get("text", "")})
        metrics = [e["text"] for e in lay["elements"] if e["testid"] == "stMetric" and y0 <= e["y"] < y1]
        (OUT / f"{out}.json").write_text(json.dumps({"aspect": (X1 - X0) / (y1 - y0), "markers": markers,
                                                     "metrics": metrics}, indent=1))
        print(f"saved {OUT / out}.png")


def main():
    if not CHROME.exists():
        print("Google Chrome not found: skip dashboard screenshots")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    port, dbg = free_port(), free_port()
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
                            "--server.headless", "true", "--server.port", str(port), "--browser.gatherUsageStats", "false",
                            "--theme.base", "light"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp = Path(tempfile.mkdtemp())
    chrome = None
    try:
        wait_http(f"http://127.0.0.1:{port}/_stcore/health")
        chrome = subprocess.Popen([str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                                   "--no-default-browser-check", f"--remote-debugging-port={dbg}", f"--user-data-dir={tmp}",
                                   f"--window-size={WIDTH},1000", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_http(f"http://127.0.0.1:{dbg}/json/version")
        pages = json.loads(wait_http(f"http://127.0.0.1:{dbg}/json/list"))
        ws_url = next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page")
        asyncio.run(capture(ws_url, f"http://127.0.0.1:{port}/"))
    finally:
        for proc in (chrome, app):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
    make_crops()


if __name__ == "__main__":
    make_crops() if "--crops-only" in sys.argv else main()
