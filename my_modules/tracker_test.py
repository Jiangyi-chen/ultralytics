from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(r"/root/Remote_venv/ultralytics/resources/models/yolo26n.pt")

video_path = "/root/Remote_venv/ultralytics/resources/videos/007.avi"
output_video_path = "007_tracked_output.mp4"


cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    result = model.track(frame, persist=True)[0]    # persist : 在连续的帧之间保持或“记住”追踪的目标

    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        frame = result.plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))

            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()