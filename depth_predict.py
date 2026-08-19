from ultralytics import YOLO
import cv2
import os

model = YOLO(r"F:\Gitbase\ultralytics\pretrain_model\yolo26n-depth.pt", task="depth")
img_path = r'F:\Gitbase\ultralytics\depth4_test.jpg'
if os.path.exists(img_path):
    img = cv2.imread(img_path)

    result = model.predict(img, save=True, project=r'F:\Gitbase\ultralytics\runs')
    print()