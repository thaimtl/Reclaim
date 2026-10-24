import sys
import time

import numpy as np
import serial

BAUD = 250000
CHANNELS = 4
FRAME = 1 + 2 * CHANNELS


def frames(port):
    with serial.Serial(port, BAUD, timeout=0.1) as s:
        time.sleep(2)
        s.reset_input_buffer()
        buf, last, seen = bytearray(), None, time.time()
        while True:
            buf += s.read(s.in_waiting or FRAME)
            rows = []
            while len(buf) >= FRAME:
                if buf[0] < 0x80 or any(b >= 0x80 for b in buf[1:FRAME]):
                    del buf[0]
                    continue
                n = buf[0] & 0x7F
                if last is not None and n != (last + 1) & 0x7F:
                    print(f"dropped {(n - last - 1) & 0x7F} frames", file=sys.stderr)
                last = n
                rows.append([buf[1 + 2 * i] << 5 | buf[2 + 2 * i] for i in range(CHANNELS)])
                del buf[:FRAME]
            if rows:
                seen = time.time()
                yield np.array(rows, dtype=np.uint16)
            elif time.time() - seen > 3:
                raise RuntimeError("no frames from the Uno for 3 s, is rig/uno/uno.ino flashed?")


if __name__ == "__main__":
    port, seconds, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    chunks, stream, t0 = [], frames(port), None
    for chunk in stream:
        t0 = t0 or time.time()
        chunks.append(chunk)
        if time.time() - t0 >= seconds:
            break
    stream.close()
    raw = np.concatenate(chunks)
    np.save(out, raw)
    print(f"saved {len(raw)} samples to {out}, {len(raw) / (time.time() - t0):.0f} Hz")
