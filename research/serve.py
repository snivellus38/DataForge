"""
Dev server for web/, with caching turned off.

`python -m http.server` sends no Cache-Control, so Chrome applies heuristic freshness and will
happily keep serving a stale field.html and field.css after an edit. That cost a round trip
already ("where is that navigation thing in the field" — it was there, the browser just had not
fetched it), and it will keep costing them, because the pages this project ships are edited far
more often than a normal static site.

So: no-store on everything, and the correct MIME type for .mjs, which the stdlib guesses wrong on
Windows often enough to break a module import with an opaque console error.

USAGE
    npm run dev            # http://localhost:8080
    python research/serve.py --port 9000
"""
import argparse
import functools
import http.server
import os
import socketserver

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--dir", default="web")
a = ap.parse_args()


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".css": "text/css",
        ".wasm": "application/wasm",
        # traces.bin.gz is inflated by the PAGE (DecompressionStream), so it must arrive as
        # opaque bytes. Naming it here keeps anything from ever attaching
        # `Content-Encoding: gzip`, which would make the browser inflate it first and hand the
        # page an already-decompressed buffer it would then fail to inflate again.
        ".gz": "application/octet-stream",
        "": "application/octet-stream",
    }

    def end_headers(self):
        # The whole point of this file.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # One line per request, without the date noise the default handler prints.
        code = args[1] if len(args) > 1 else ""
        if str(code).startswith(("4", "5")):
            print(f"  {code}  {args[0]}")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


root = os.path.abspath(a.dir)
if not os.path.isdir(root):
    raise SystemExit(f"no such directory: {root}")

handler = functools.partial(Handler, directory=root)
with Server(("", a.port), handler) as httpd:
    print(f"serving {root}")
    print(f"  http://localhost:{a.port}/            -> redirects to the field")
    print(f"  http://localhost:{a.port}/field.html  THE FIELD")
    print(f"  http://localhost:{a.port}/loop.html   THE LOOP")
    print(f"  http://localhost:{a.port}/price.html  THE PRICE")
    print("caching disabled — a plain reload always gets the current bytes. ctrl-c to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
