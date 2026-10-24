"""Smoke-test the camera chain before real recordings exist: cut any face video into clips and run everything.

    bash third_party/lipflow/scripts/download-models.sh --samples   # public-domain sample video (optional)
    venv-camera/bin/python -m camera.smoke third_party/lipflow/samples/2016-03-12.mov

Builds recordings_smoke/s0/manifest.jsonl (sets C and E alternate; prompts are placeholders, so WER is meaningless
here), then runs camera.run_camera_baselines on it. Checks that every system loads, decodes, and writes results.
"""
import json
import os
import subprocess
import sys

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo


def main():
    video = os.path.abspath(sys.argv[1])
    clips = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = os.path.join(WORK, 'recordings_smoke', 's0')
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = []
    for i in range(clips):
        name = f'smoke_{i:02d}.mp4'
        writer = cv2.VideoWriter(os.path.join(out, name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for _ in range(int(4 * fps)):  # 4 s per clip
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        rows.append({'id': f'smoke_{i:02d}', 'session': 's0', 'set': 'CE'[i % 2], 'prompt': 'placeholder words only',
                     'target_wpm': [100, 150, 180][i % 3], 'mouth': 'open', 'voiced': True, 'emg': None,
                     'video': name, 'video_ts': None, 't_start': None, 't_end': None})
    with open(os.path.join(out, 'manifest.jsonl'), 'w') as f:
        f.write('\n'.join(json.dumps(r) for r in rows) + '\n')
    subprocess.run([sys.executable, '-m', 'camera.run_camera_baselines', '--systems', 'auto_avsr,braven',
                    '--recordings', os.path.dirname(out)], cwd=ROOT, check=True)
    print('CAMERA SMOKE PASS: systems ran end to end (ignore WER: prompts are placeholders)')


if __name__ == '__main__':
    main()
