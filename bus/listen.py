"""Print what the camera reader publishes: one line per message, validated, with bus latency (now - t_pub).

    venv-live/bin/python -m bus.listen [--sub ipc:///tmp/reclaim/cam.tick] [--topic cam.face] [--duration 30]

Ends with a summary: messages per topic, invalid ones, seq gaps (dropped messages), publisher restarts (seq went
back), payload size, bus latency p50/p95, and the camera's own pass time (timing_ms total) p50/p95.
"""
import argparse

import numpy as np

from bus import CAM_PUB, FACE, FINAL, Subscriber, TICK, now, decode, validate


def summary_line(topic, msg, lat_ms):
    if topic == TICK:
        T = len(msg['frame_t'])
        e = msg['emotion']
        return (f'{topic} seq={msg["seq"]} seg={msg["seg_id"]} T={T} face={int(msg["face_ok"])} '
                f'act={float(msg["activity"][-1]):.2f} emo={e["label"]}:{e["confidence"]:.2f} '
                f'pass={msg["timing_ms"]["total"]:.1f}ms lat={lat_ms:.2f}ms  {msg["greedy"]!r}')
    if topic == FACE:
        pose = msg['pose']
        return (f'{topic} seq={msg["seq"]} face={int(msg["face_ok"])} '
                + (f'pts={msg["pts"].shape[0]} ' if msg['face_ok'] else '')
                + (f'yaw={np.degrees(pose[0]):+.0f} pitch={np.degrees(pose[1]):+.0f} roll={np.degrees(pose[2]):+.0f} '
                   if pose is not None else '') + f'lat={lat_ms:.2f}ms')
    best = msg['nbest'][0]
    return f'{topic} seq={msg["seq"]} seg={msg["seg_id"]} n={len(msg["nbest"])} best={best["text"]!r} ({best["score"]})'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sub', default=CAM_PUB)
    ap.add_argument('--topic', action='append', default=None,
                    help='topic prefix (repeatable; default cam.tick + cam.final; cam.face or cam. for faces too)')
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--quiet', action='store_true', help='summary only')
    args = ap.parse_args()

    sub = Subscriber(args.sub, topics=args.topic or (TICK, FINAL))
    counts, bad, gaps, restarts, last_seq, lat, size, pass_ms = {}, 0, 0, 0, {}, [], [], []
    start = now()
    print(f'listening on {args.sub}', flush=True)
    try:
        while args.duration is None or now() - start < args.duration:
            raw = sub.sock.poll(200)
            if not raw:
                continue
            frames = sub.sock.recv_multipart()
            t_rx = now()
            size.append(len(frames[-1]))
            try:
                topic, msg = decode(frames)
                validate(topic, msg)
            except ValueError as e:
                bad += 1
                print(f'INVALID {frames[0][:20]!r}: {e}', flush=True)
                continue
            counts[topic] = counts.get(topic, 0) + 1
            if topic in last_seq and msg['seq'] <= last_seq[topic]:
                restarts += 1
                print(f'{topic}: seq {last_seq[topic]} -> {msg["seq"]}, publisher restarted', flush=True)
            elif topic in last_seq:
                gaps += msg['seq'] - last_seq[topic] - 1
            last_seq[topic] = msg['seq']
            lat.append((t_rx - msg['t_pub']) * 1e3)
            if topic == TICK:
                pass_ms.append(msg['timing_ms']['total'])
            if not args.quiet:
                print(summary_line(topic, msg, lat[-1]), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
    if lat:
        print(f'summary: {counts} invalid={bad} dropped={gaps} restarts={restarts} payload max={max(size) / 1e3:.1f} KB '
              f'lat p50={np.percentile(lat, 50):.3f} ms p95={np.percentile(lat, 95):.3f} ms'
              + (f' pass p50={np.percentile(pass_ms, 50):.1f} ms p95={np.percentile(pass_ms, 95):.1f} ms' if pass_ms else ''),
              flush=True)


if __name__ == '__main__':
    main()
