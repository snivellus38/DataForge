#!/usr/bin/env python3
"""Narrow-viewport check: do any of the four pages overflow horizontally on a phone?

    npm run test:mobile                    # 390 and 360, all four pages
    python research/check_mobile.py --widths 320 --pages loop.html

This is NOT part of `npm test`, deliberately. The twelve gates run in jsdom, which has no layout
engine and cannot answer this question at all; this one needs a real browser and a running
server, and a unit suite should not depend on either. It is a pre-release check.

Requires: `pip install websocket-client`, and Edge or Chrome. Starts its own dev server and its
own headless browser, and cleans both up.

Three things this exists to remember, all of which cost time on 2026-09-06:

1. A page can look completely broken and be fine. `msedge --headless=old --window-size=390,N
   --screenshot` renders and captures at different widths, so it produced a clipped image of a
   page whose layout was correct. Measure with getBoundingClientRect over CDP; screenshot only to
   judge whether the result reads well.

2. `overflow-x: hidden` on html/body is not a fix, it is a gag. It makes documentElement
   .scrollWidth equal the viewport no matter how far elements stick out, so this check would pass
   on a broken page. That is why the report counts OFFENDING ELEMENTS as well as scroll width.

3. Overflow contained by an ancestor with overflow-x auto/scroll/hidden is legitimate -- a wide
   code block or a diagram that scrolls sideways on purpose. Those are listed as "contained" and
   do not fail the check. Only overflow that reaches the document fails it.

4. scrollWidth vs innerWidth is NOT sufficient, and this is the subtle one. When content cannot
   fit, the browser widens the layout viewport and zooms the page out instead of overflowing, so
   innerWidth grows to match and the two agree perfectly on a page rendering at 82%. loop.html
   laid out at 473px on a 390px screen with every label shrunk to mush, and reported "ok". Both
   numbers are therefore compared against the REQUESTED device width, not against each other.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

try:
    import websocket  # websocket-client
except ImportError:
    sys.exit("needs websocket-client:  pip install websocket-client")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

PROBE = r"""
(() => {
  const vw = window.innerWidth;
  const loose = [], held = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (r.right <= vw + 1) continue;
    let a = el.parentElement, box = null;
    while (a && a !== document.documentElement) {
      if (/auto|scroll|hidden/.test(getComputedStyle(a).overflowX)) {
        box = a.tagName.toLowerCase() + (a.id ? '#' + a.id : '');
        break;
      }
      a = a.parentElement;
    }
    const row = {
      sel: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
           (el.className ? '.' + String(el.className).trim().split(/\s+/)[0] : ''),
      right: Math.round(r.right), box,
    };
    (box ? held : loose).push(row);
  }
  return JSON.stringify({
    vw, scrollW: document.documentElement.scrollWidth,
    loose: loose.sort((a, b) => b.right - a.right).slice(0, 6),
    nHeld: held.length,
  });
})()
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", nargs="*", type=int, default=[390, 360])
    ap.add_argument("--pages", nargs="*",
                    default=["index.html", "field.html", "loop.html", "price.html"])
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--cdp", type=int, default=9782)
    a = ap.parse_args()

    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        sys.exit("no Edge or Chrome found")

    server = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "serve.py"), "--port", str(a.port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # --headless=new, not old: old headless does not expose the DevTools endpoint here.
    proc = subprocess.Popen(
        [browser, "--headless=new", "--disable-gpu", f"--remote-debugging-port={a.cdp}",
         "--remote-allow-origins=*",
         # a temp profile, not one under research/: the browser holds file locks on it
         # for a while after exit, so a repo-local one lingers as undeletable clutter
         "--user-data-dir=" + os.path.join(tempfile.gettempdir(), "bdh-mobile-profile"),
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    ws = None
    try:
        ws_url = None
        for _ in range(120):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.cdp}/json"))
                ws_url = next((t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page"),
                              None)
                if ws_url:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not ws_url:
            sys.exit("browser never exposed a debug target")

        ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        n = [0]

        def send(method, params=None):
            n[0] += 1
            ws.send(json.dumps({"id": n[0], "method": method, "params": params or {}}))
            while True:
                m = json.loads(ws.recv())
                if m.get("id") == n[0]:
                    return m

        send("Page.enable")
        send("Network.enable")
        # This browser outlives edits during a session. Without this it replays a cached
        # stylesheet and reports the OLD layout -- which looks exactly like "my fix did nothing".
        send("Network.setCacheDisabled", {"cacheDisabled": True})

        bad = 0
        for width in a.widths:
            send("Emulation.setDeviceMetricsOverride",
                 {"width": width, "height": 900, "deviceScaleFactor": 1, "mobile": True})
            print(f"\n── {width}px ─────────────────────────────────────────────")
            for page in a.pages:
                send("Page.navigate", {"url": f"http://localhost:{a.port}/{page}"})
                time.sleep(5.0)
                d = json.loads(send("Runtime.evaluate",
                                    {"expression": PROBE, "returnByValue": True})
                               ["result"]["result"]["value"])
                # Two distinct failures, and the second one is the sneaky one.
                #  - overflow: the layout is wider than the viewport, so the page scrolls
                #    sideways.
                #  - SHRINK-TO-FIT: the content could not fit, so the browser widened the layout
                #    viewport and zoomed the whole page out. innerWidth then EXCEEDS the device
                #    width while scrollWidth still equals innerWidth -- so comparing those two
                #    reports "ok" on a page rendering at 82% with unreadable labels. Measured:
                #    loop.html at a 390px device laid out at 473 and passed that way.
                zoomed = d["vw"] > width + 1
                over = d["scrollW"] - d["vw"]
                fail = over > 1 or zoomed or d["loose"]
                bad += bool(fail)
                held = f", {d['nHeld']} contained" if d["nHeld"] else ""
                note = (f"  SHRUNK TO FIT: laid out at {d['vw']}px on a {width}px screen"
                        if zoomed else "")
                print(f"  {'FAIL' if fail else 'ok  '}  {page:12s} "
                      f"layout {d['scrollW']:4d} vs {width}{held}{note}")
                for r in d["loose"]:
                    print(f"          overflows: {r['sel']}  right {r['right']}")
        print()
        print("MOBILE FAILED" if bad else "MOBILE OK — no page overflows its viewport")
        return 1 if bad else 0
    finally:
        if ws:
            ws.close()
        proc.terminate()
        server.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
