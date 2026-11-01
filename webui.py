import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = (Path(__file__).parent / "web").resolve()
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


def serve(routes, port, page, open_browser=True):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, method, body):
            path = self.path.split("?")[0]
            fn = routes.get((method, path))
            if fn:
                try:
                    out = fn(body)
                    return self._send(200, out if isinstance(out, str) else json.dumps(out))
                except (AssertionError, ValueError, KeyError) as e:
                    return self._send(400, json.dumps({"error": str(e)}))
            f = (WEB / (page if path == "/" else path.lstrip("/"))).resolve()
            if method == "GET" and f.is_file() and WEB in f.parents:
                return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
            self._send(404, json.dumps({"error": "not found"}))

        def do_GET(self):
            self._route("GET", None)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self._route("POST", json.loads(self.rfile.read(n) or b"{}"))

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        allow_reuse_address = os.name != "nt"

    try:
        srv = Server(("127.0.0.1", port), Handler)
    except OSError:
        sys.exit(f"Port {port} is already in use. Close the other copy that is running, or pick another with --port.")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"Dashboard: {url}")
    if open_browser:
        webbrowser.open(url)


class Dashboard:
    def __init__(self, title, port, open_browser=True, options=None):
        self.lock = threading.Lock()
        self.start = time.time()
        self.options = dict(options or {})
        self.state = {"title": title, "status": "starting", "progress": 0, "charts": {},
                      "log": [], "hand": None, "network": None, "stats": {}, "options": dict(self.options)}
        serve({("GET", "/api/state"): self._get, ("POST", "/api/options"): self._set_options},
              port, "train.html", open_browser)

    def _set_options(self, body):
        with self.lock:
            for key, value in body.items():
                if key not in self.options:
                    raise KeyError(f"unknown option {key}")
                self.options[key] = type(self.options[key])(value)
            self.state["options"] = dict(self.options)
            return {"options": self.options}

    def _get(self, _):
        with self.lock:
            self.state["elapsed"] = round(time.time() - self.start)
            return json.dumps(self.state)

    def set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def point(self, chart, series, x, y, log=False):
        with self.lock:
            c = self.state["charts"].setdefault(chart, {"series": {}, "log": log})
            pts = c["series"].setdefault(series, [])
            pts.append([x, y])
            if len(pts) > 1500:
                c["series"][series] = pts[::2]

    def log(self, msg):
        print(msg)
        with self.lock:
            self.state["log"] = (self.state["log"] + [f"[{time.strftime('%H:%M:%S')}] {msg}"])[-200:]
