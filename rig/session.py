import shutil
import subprocess
import sys
import time

import numpy as np
from serial.tools import list_ports

from filters import EmgFilter
from reader import CHANNELS, frames

STEPS = [("rest", 5), ("pa pa pa", 10), ("rest", 5), ("ta ta ta", 10), ("rest", 5), ("I'm thirsty", 10)]
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86}


def say(text):
    print(f"\n>>> {text.upper()}")
    if shutil.which("say"):
        subprocess.Popen(["say", text])


def find_port():
    ports = [p.device for p in list_ports.comports() if p.vid in ARDUINO_VIDS]
    if not ports:
        sys.exit("Uno not found. Plug in its USB cable, or pass the port as the second argument.")
    return ports[0]


def run_for(stream, seconds, on_chunk=None):
    end = time.time() + seconds
    for chunk in stream:
        if on_chunk:
            on_chunk(chunk, end)
        if time.time() >= end:
            return


def main(out, port=None):
    port = port or find_port()
    input(f"Uno on {port}. Pads on and relaxed? Press Enter to start. ")
    stream, live = frames(port), EmgFilter()
    next(stream)
    for n in ("3", "2", "1"):
        say(n)
        run_for(stream, 1)
    raws, labels, recent = [], [], np.zeros((0, CHANNELS))
    for label, seconds in STEPS:
        say(label)

        def keep(chunk, end):
            nonlocal recent
            raws.append(chunk)
            labels.extend([label] * len(chunk))
            recent = np.vstack([recent, live(chunk)])[-150:]
            bars = "  ".join(f"A{c} {'#' * min(10, int(recent[:, c].std())):<10}" for c in range(CHANNELS))
            print(f"\r{max(0, end - time.time()):4.1f}s  {bars}", end="", flush=True)

        run_for(stream, seconds, keep)
    stream.close()
    np.savez(out, raw=np.concatenate(raws), step=np.array(labels))
    say("done")
    print(f"Saved {out}. Check it with: python rig/check.py {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
