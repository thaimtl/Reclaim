"""The live dashboard: every team-bus topic -> CallState -> the page (Thai's Figma v5), over one websocket.

    venv-call/bin/python -m dashboard.server [--port 8790] [--number "(628) 888-6598"]
    then open http://127.0.0.1:8790 full screen (F11). Replay a logged call with buslog.replay, or rehearse with
    python -m dashboard.demo_feed; the dashboard can't tell them from a live call.

It only listens (SUB to every endpoint in bus/team.py), so it can start and stop at any time without touching
the call. Snapshots go out SNAPSHOT_HZ times a second as JSON text frames; cam.preview JPEGs (the lip reader's
mirror of the webcam) go out as binary frames, the newest one at most PREVIEW_HZ times a second, so a slow browser
gets fewer frames, never a backlog.
"""
import argparse
import asyncio
import json
import logging
import os
import time

import zmq
import zmq.asyncio
from aiohttp import WSMsgType, web

import bus
from bus.team import ENDPOINTS, decode
from buslog import log_root
from dashboard.state import CallState

log = logging.getLogger('dashboard')
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, 'static')
TOPICS = ('call.state', 'partner.text', 'dec.', 'claude.repair', 'voice.say', 'cam.tick', 'cam.face',
          'cam.preview', 'emg.raw', 'timing.')
SNAPSHOT_HZ = 15
PREVIEW_HZ = 15
LOG_FRESH_S = 5.0   # the logger flushes every 0.2 s: older than this, it isn't running


def load_benchmark(path=None):
    """Word error per reader on a scored session, in %, or None (the end screen then shows its placeholder)."""
    if path is None:
        work = os.environ.get('RECLAIM_WORKDIR')
        path = os.path.join(work or '.', 'results', 'wer_session3.json')
    try:
        with open(path) as f:
            b = json.load(f)
        return {k: float(b[k]) for k in ('lips', 'lips_muscles', 'claude')} | {'session': str(b.get('session', ''))}
    except (OSError, ValueError, KeyError, TypeError):
        return None


class Dashboard:
    def __init__(self, number='', endpoints=None, ctx=None, benchmark=None):
        self.state = CallState(number, benchmark)
        self.endpoints = dict(endpoints or ENDPOINTS)
        self.ctx = ctx or zmq.asyncio.Context.instance()
        self.clients = set()
        self.preview = None              # newest cam.preview JPEG bytes, sent once to each client
        self.preview_seq = 0
        self.tasks = []

    def subscribe(self):
        sock = self.ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 2000)
        for addr in self.endpoints.values():
            sock.connect(addr)
        for t in TOPICS:
            sock.setsockopt(zmq.SUBSCRIBE, t.encode())
        return sock

    def handle(self, frames):
        """One bus message; a malformed one is skipped (the dashboard never stops the call)."""
        try:
            topic, msg = decode(frames, validate=False)
        except ValueError:
            return
        if topic == 'cam.preview':
            jpeg = msg.get('jpeg')
            if isinstance(jpeg, (bytes, bytearray)) and jpeg:
                self.preview, self.preview_seq = bytes(jpeg), self.preview_seq + 1
            return
        try:
            self.state.feed(topic, msg, bus.now())
        except Exception:  # noqa: BLE001  one odd message must not stop the screen
            log.exception('bad %s message', topic)

    async def bus_loop(self):
        sock = self.subscribe()
        try:
            while True:
                self.handle(await sock.recv_multipart())
        finally:
            sock.close(linger=0)

    def check_logging(self):
        """The bus logger (buslog.record) wrote within the last few seconds: its newest file is fresh."""
        try:
            sessions = sorted(log_root().iterdir())
            newest = max((f.stat().st_mtime for f in sessions[-1].glob('bus-*.msgpack')), default=0) if sessions else 0
        except OSError:
            return
        if time.time() - newest < LOG_FRESH_S:
            self.state.logging(bus.now())

    async def push_loop(self):
        sent_preview, period, k = 0, 1.0 / SNAPSHOT_HZ, 0
        while True:
            await asyncio.sleep(period)
            k += 1
            if k % SNAPSHOT_HZ == 0:
                self.check_logging()
            if not self.clients:
                continue
            try:
                text = json.dumps(self.state.snapshot(bus.now()), separators=(',', ':'))
            except Exception:  # noqa: BLE001  a bad state skips one frame; the screen keeps updating
                log.exception('snapshot failed')
                continue
            preview = self.preview if self.preview_seq != sent_preview and k % max(1, SNAPSHOT_HZ // PREVIEW_HZ) == 0 \
                else None
            if preview is not None:
                sent_preview = self.preview_seq
            for ws in list(self.clients):
                try:
                    await ws.send_str(text)
                    if preview is not None:
                        await ws.send_bytes(preview)
                except (ConnectionError, RuntimeError):
                    self.clients.discard(ws)

    async def websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            async for m in ws:
                if m.type == WSMsgType.ERROR:
                    break
        finally:
            self.clients.discard(ws)
        return ws

    async def index(self, request):
        return web.FileResponse(os.path.join(STATIC, 'index.html'))

    def app(self):
        app = web.Application()
        app.router.add_get('/', self.index)
        app.router.add_get('/ws', self.websocket)
        app.router.add_static('/static/', STATIC)

        async def start(app):
            self.tasks = [asyncio.create_task(self.bus_loop()), asyncio.create_task(self.push_loop())]

        async def stop(app):
            for t in self.tasks:
                t.cancel()
            for ws in list(self.clients):
                await ws.close()
        app.on_startup.append(start)
        app.on_cleanup.append(stop)
        return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=8790)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--number', default='', help="the judge's number as shown in the header, e.g. '(628) 888-6598'")
    ap.add_argument('--benchmark', help='scored word error JSON (default: $RECLAIM_WORKDIR/results/wer_session3.json)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    print(f'dashboard: http://{args.host}:{args.port}')
    benchmark = load_benchmark(args.benchmark)
    print('end screen word error:', f"from {benchmark['session'] or 'the benchmark file'}" if benchmark else 'placeholder')
    web.run_app(Dashboard(args.number, benchmark=benchmark).app(), host=args.host, port=args.port, print=None,
                access_log=None)


if __name__ == '__main__':
    main()
