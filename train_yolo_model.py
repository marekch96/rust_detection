from  ultralytics import YOLO
import torch

import cv2


#print(torch.cuda.is_available())
model = YOLO('yolo11l-seg.yaml')

results= model.train(data='config.yaml',epochs=100,batch=8,name='rust_detection_new_dataset_seg_b8_e100')