# Reclaim dashboard (design prototype)

Run with the live webcam:

    cd demo/dashboard && python3 -m http.server 8000

Open http://localhost:8000/?cam in Chrome on the same machine (the camera needs localhost or https).
Without ?cam it plays recorded face-tracking data; click "Use webcam" to switch. Press H for "Under the hood".

The face is MediaPipe Face Landmarker, the same model Lipflow uses for mouth crops: 468 landmarks, blendshapes, head pose.
Call content, timings and EMG levels are still sample data.

track_face.py VIDEO face_landmarker.task OUT.json START DUR FPS turns a clip of Austin into playback data.
