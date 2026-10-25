"""cam.preview: the dashboard's mirror of the webcam, which only the lip reader can open.

    preview = PreviewWorker(lambda m: pub.send(m, block=False))
    preview.submit(frame.t, frame.image, cropper.last_face, cropper.last_z)    # the crop thread, every frame
    preview.close()

The crop thread only hands over a reference to the newest frame (submit() never waits and never copies); this worker
thread shrinks it to WIDTH px, JPEG-encodes it (OpenCV releases the GIL) and sends it without blocking, at most HZ
times a second. A frame it hasn't reached yet is replaced by the next one, so a slow encode or a busy socket costs
preview frames, never lip-reading time. Each message carries that frame's own face points, so the dashboard draws the
pads on the very frame it shows.
"""
import logging
import threading
import time

import cv2

from bus import preview_msg

log = logging.getLogger(__name__)

HZ = 30
WIDTH = 640
QUALITY = 80


class PreviewWorker:
    def __init__(self, publish, hz=HZ, width=WIDTH, quality=QUALITY):
        self.publish, self.period, self.width, self.quality = publish, 1.0 / hz, width, quality
        self.newest = None                  # (t, image, last_face, last_z) of the newest frame not sent yet
        self.slot = threading.Lock()        # guards newest: a swap never loses a frame submitted mid-swap
        self.ready = threading.Event()
        self.stopped = False
        self.sent = self.errors = 0
        self.thread = threading.Thread(target=self._run, name='preview', daemon=True)
        self.thread.start()

    def submit(self, t, image, last_face, last_z):
        """The crop thread's newest frame and its face (LiveCropper.last_face / last_z). Never waits on the encode."""
        with self.slot:
            self.newest = (t, image, last_face, last_z)
        self.ready.set()

    def _run(self):
        last = 0.0
        while True:
            self.ready.wait()
            if self.stopped:
                return
            time.sleep(max(0.0, last + self.period - time.perf_counter()))
            self.ready.clear()
            if self.stopped:                # close() came during the wait: its wake-up was just cleared
                return
            with self.slot:
                item, self.newest = self.newest, None
            if item is None:
                continue
            last = time.perf_counter()
            try:
                self._send(*item)
            except Exception:  # noqa: BLE001  (the preview is optional: it never stops the lip reader)
                self.errors += 1
                if self.errors == 1:
                    log.exception('cam.preview failed (later failures are only counted)')

    def _send(self, t, image, last_face, last_z):
        h, w = image.shape[:2]
        small = image if w <= self.width else cv2.resize(image, (self.width, round(h * self.width / w)),
                                                         interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise ValueError('JPEG encode failed')
        if last_face is not None and last_face[0] != t:     # the face of another frame: don't draw it on this one
            last_face = last_z = None
        if self.publish(preview_msg(t, jpeg.tobytes(), w, h, last_face, last_z)):
            self.sent += 1

    def close(self):
        self.stopped = True
        self.ready.set()
        self.thread.join(timeout=1.0)
        if self.errors:
            log.warning('cam.preview failed on %d frames', self.errors)
