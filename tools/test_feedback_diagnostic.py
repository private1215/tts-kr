"""隔离验证真实反馈API函数；--serve 提供支持 Range 的浏览器测试页。"""
import argparse
import ast
import asyncio
import json
import os
import pathlib
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs/feedback_20261013b_diag"


def api_test():
    source = ast.parse((ROOT / "app/server.py").read_text())
    nodes = [n for n in source.body if isinstance(n, ast.AsyncFunctionDef)
             and n.name in ("fb_vote", "fb_votes", "fb_items")]
    for n in nodes:
        n.decorator_list = []
        n.returns = None
        for arg in n.args.args:
            arg.annotation = None
    class HTTPException(Exception):
        def __init__(self, status, detail):
            self.status = status
    class Request:
        def __init__(self, body):
            self.body = body
        async def json(self):
            return self.body
    with tempfile.TemporaryDirectory() as tmp:
        scope = dict(os=os, json=json, time=time, FB_DIR=tmp,
                     FB_VOTES=tmp + "/votes.jsonl", HTTPException=HTTPException,
                     current_user=lambda r: {"username": "isolated_test"},
                     require_admin=lambda r: None,
                     JSONResponse=lambda body, **kw: body)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "feedback_api", "exec"), scope)
        config = json.loads((BUNDLE / "items_20261013b_diag.json").read_text())
        pathlib.Path(tmp, "items.json").write_text(json.dumps(config))
        assert asyncio.run(scope["fb_items"]()) == {"items": config["items"]}
        ident = config["items"][0]["id"]
        for key in ("A", "B", "C", "D", "E"):
            assert asyncio.run(scope["fb_vote"](Request({"picks": {ident: key}, "free": "原音频秒=192.400"}))) == {"ok": True}
        raw = pathlib.Path(tmp, "votes.jsonl").read_text().splitlines()
        assert len(raw) == 5
        assert [json.loads(r)["picks"][ident] for r in raw] == list("ABCDE")
        assert asyncio.run(scope["fb_votes"](Request({})))["by_item"][ident] == {"E": 1}
        print("通过：实际API热读、A/B/C/D/E独立保存、append-only、去重、时间点文本保存；生产votes未写入")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/me":
            self.send_response(401)
            self.end_headers()
            return
        name = self.path.split("?")[0].split("/")[-1]
        path = BUNDLE / ("items_20261013b_diag.json" if self.path == "/api/fb/items"
                         else "feedback.html" if self.path == "/feedback" else name)
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        start, end = 0, len(data) - 1
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.removeprefix("bytes=").split("-", 1)
            start, end = int(a), min(int(b) if b else end, end)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "audio/mpeg" if name.endswith("mp3") else "text/html; charset=utf-8")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, len(data)))
        self.end_headers()
        self.wfile.write(data[start:end + 1])

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert "picks" in data and len(data["free"]) <= 4000
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    api_test()
    if args.serve:
        ThreadingHTTPServer(("127.0.0.1", 8769), Handler).serve_forever()
