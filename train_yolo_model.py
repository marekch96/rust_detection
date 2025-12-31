from  ultralytics import YOLO
import torch

import cv2


print(torch.cuda.is_available())
#model = YOLO('yolo11l.yaml')

#results= model.train(data='config.yaml',epochs=50,batch=8,name='rust_detector_v11l')