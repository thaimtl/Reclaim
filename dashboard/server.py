"""The live dashboard: every team-bus topic -> CallState -> the page (Thai's Figma v5), over one websocket.

    venv-call/bin/python -m dashboard.server [--port 8790] [--number "(628) 888-6598"]
    then open http://127.0.0.1:8790 full screen (F11). Replay a logged call with buslog.replay, or rehearse with
    python -m dashboard.demo_feed; the dashboard can't tell them from a live call.

It only listens (SUB to every endpoint in bus/team.py), so it can start and stop at any time without touching
the call. Snapshots go out SNAPSHOT_HZ times a second as JSON text frames. Each cam.preview JPEG (the lip reader's
mirror of the webcam) goes out the moment it arrives, as one binary frame with the face points of that same video
frame (FRAME_MAGIC, header length, JSON header {t, w, h, pts}, JPEG), so the page draws the picture and the pads on
it in the same paint: the pads stay on his face like a camera filter, never a frame behind. Points come from the
preview message itself if the camera reader puts them there, else from the cam.face nearest in time. A browser
that is slower than the camera gets the newest frame, never a backlog.
"""
import argparse
import asyncio
import json
import logging
import os
import struct
import time
from collections import deque

import numpy as np
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
FRAME_MAGIC = b'RCF1'   # binary frame: magic, 4-byte big-endian header length, JSON header, JPEG
FACE_MATCH_S = 0.06     # a cam.face this close in time is the preview frame's own (the camera runs at ~12.5 Hz)
LOG_FRESH_S = 5.0   # the logger flushes every 0.2 s: older than this, it isn't running


class Dashboard:
    def __init__(self, number='', endpoints=None, ctx=None):
        self.state = CallState(number)
        self.endpoints = dict(endpoints or ENDPOINTS)
        self.ctx = ctx or zmq.asyncio.Context.instance()
        self.clients = set()
        self.frame = None                # newest binary frame for the page: the JPEG and its face points
        self.viewers = {}                # websocket -> [newest frame not sent yet, asyncio.Event]: one sender each
        self.faces = deque(maxlen=40)    # recent cam.face (t, pts): each preview frame takes its own
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
                try:
                    self.frame = pack_frame(msg, self.face_for(msg))
                except (TypeError, ValueError):         # malformed points: the picture alone
                    self.frame = pack_frame(msg, None)
                for slot in self.viewers.values():
                    slot[0] = self.frame
                    slot[1].set()
            return
        if topic == 'cam.face' and msg.get('face_ok') and msg.get('pts') is not None:
            try:
                self.faces.append((float(msg['t']), points(msg['pts'])))
            except (KeyError, TypeError, ValueError):   # malformed: not kept
                pass
        try:
            self.state.feed(topic, msg, bus.now())
        except Exception:  # noqa: BLE001  one odd message must not stop the screen
            log.exception('bad %s message', topic)

    def face_for(self, msg):
        """The face points of a preview frame: its own (if the camera reader sends them), else the cam.face closest
        in time within FACE_MATCH_S, else None (the page then keeps its last placement)."""
        if msg.get('pts') is not None:
            return points(msg['pts'])
        try:
            t = float(msg['t'])
        except (KeyError, TypeError, ValueError):
            return None
        best = min(self.faces, key=lambda f: abs(f[0] - t), default=None)
        return best[1] if best is not None and abs(best[0] - t) <= FACE_MATCH_S else None

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

    async def send_frames(self, ws, slot):
        """One page's sender: the newest frame, as soon as there is one, so a slow page skips frames and never holds up
        another page."""
        while True:
            await slot[1].wait()
            slot[1].clear()
            frame, slot[0] = slot[0], None
            if frame is not None:
                await ws.send_bytes(frame)

    async def push_loop(self):
        period, k = 1.0 / SNAPSHOT_HZ, 0
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
            for ws in list(self.clients):
                try:
                    await ws.send_str(text)
                except (ConnectionError, RuntimeError):
                    self.clients.discard(ws)

    async def websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=10, compress=False)     # JPEGs are compressed already
        await ws.prepare(request)
        self.clients.add(ws)
        slot = self.viewers[ws] = [self.frame, asyncio.Event()]
        if self.frame is not None:
            slot[1].set()
        sender = asyncio.create_task(self.send_frames(ws, slot))
        try:
            async for m in ws:
                if m.type == WSMsgType.ERROR:
                    break
        finally:
            sender.cancel()
            self.clients.discard(ws)
            self.viewers.pop(ws, None)
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


def points(pts):
    """Face points as (N, 2) floats; ValueError if they aren't."""
    return np.asarray(pts, np.float64).reshape(-1, 2)


def pack_frame(msg, pts):
    """A preview frame for the page: FRAME_MAGIC, header length, JSON header (t, w, h, pts rounded), JPEG."""
    head = json.dumps({'t': msg.get('t'), 'w': msg.get('w'), 'h': msg.get('h'),
                       'pts': None if pts is None else np.round(points(pts), 4).tolist()},
                      separators=(',', ':')).encode()
    return FRAME_MAGIC + struct.pack('>I', len(head)) + head + bytes(msg['jpeg'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=8790)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--number', default='', help="the judge's number as shown in the header, e.g. '(628) 888-6598'")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    print(f'dashboard: http://{args.host}:{args.port}')
    web.run_app(Dashboard(args.number).app(), host=args.host, port=args.port, print=None, access_log=None)


if __name__ == '__main__':
    main()
