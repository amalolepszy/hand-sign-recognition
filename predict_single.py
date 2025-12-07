
"""
predict_single.py
Simple inference script using ultralytics YOLOv8
python predict_single.py --model runs/detect/train/weights/best.pt --image /path/to/img.jpg
"""
import argparse
from ultralytics import YOLO
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--conf", type=float, default=0.35)
args = parser.parse_args()

model = YOLO(args.model)
res = model.predict(source=args.image, conf=args.conf, save=True)
print("Saved predictions to runs/detect/predict/")
