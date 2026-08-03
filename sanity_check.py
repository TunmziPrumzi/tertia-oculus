from ultralytics import YOLO
import torch

print("Torch CUDA available:", torch.cuda.is_available())

model = YOLO("yolov8n.pt")
print(model.info())