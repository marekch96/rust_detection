from  ultralytics import YOLO
import cv2

model = YOLO('yolo11l.yaml')

results= model.train(data='config.yaml',epochs=50,batch=8,name='rust_detector_v11l')