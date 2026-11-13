"""Quick webcam check - NO Kivy. Run:  python cam_test.py
Tells you which camera index/backend actually returns frames."""
import os
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_MSMF', '0')

import cv2

backends = [('default', None), ('DSHOW', getattr(cv2, 'CAP_DSHOW', None)),
            ('MSMF', getattr(cv2, 'CAP_MSMF', None))]

for index in range(3):
    for name, api in backends:
        if name != 'default' and api is None:
            continue
        cap = cv2.VideoCapture(index) if api is None else cv2.VideoCapture(index, api)
        ok, frame = (False, None)
        if cap.isOpened():
            for _ in range(15):          # warm-up frames
                ok, frame = cap.read()
                if ok:
                    break
        shape = None if frame is None else frame.shape
        print(f'index {index}  backend {name:8s} opened={cap.isOpened()}  frame_ok={ok}  shape={shape}')
        cap.release()
