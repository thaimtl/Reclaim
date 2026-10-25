"""Run MediaPipe Face Landmarker on a clip; write landmarks, blendshapes and head pose per frame (for the dashboard mockup).
Usage: python track_face.py VIDEO MODEL.task OUT.json [start_s] [dur_s] [fps]"""
import sys, json, math
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

video, model, out = sys.argv[1:4]
start, dur, fps = (float(a) for a in (sys.argv[4:7] + ['0', '9999', '15'][len(sys.argv[4:7]):]))
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1))
cap = cv2.VideoCapture(video); src_fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
keep = ['jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
        'mouthPucker', 'eyeBlinkLeft', 'eyeBlinkRight']
frames, t, step = [], 0.0, 1.0 / fps
while t < dur:
    cap.set(cv2.CAP_PROP_POS_MSEC, (start + t) * 1000)
    ok, im = cap.read()
    if not ok: break
    h, w = im.shape[:2]
    r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), int((start + t) * 1000))
    t += step
    if not r.face_landmarks: continue
    P = np.array([[p.x * w, p.y * h, p.z * w] for p in r.face_landmarks[0][:468]])
    bs = {c.category_name: c.score for c in r.face_blendshapes[0]}
    M = np.array(r.facial_transformation_matrixes[0])[:3, :3]
    yaw = math.atan2(-M[2, 0], math.hypot(M[2, 1], M[2, 2])); pitch = math.atan2(M[2, 1], M[2, 2]); roll = math.atan2(M[1, 0], M[0, 0])
    frames.append({'p': P.tolist(), 'b': [round(bs.get(k, 0), 3) for k in keep], 'pose': [round(yaw, 3), round(pitch, 3), round(roll, 3)]})
json.dump({'fps': fps, 'keys': keep, 'frames': frames}, open(out, 'w'))
print(len(frames), 'frames')
