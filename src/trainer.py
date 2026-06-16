import argparse, wandb, datetime
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--data_yaml", type=str, required=True)
parser.add_argument("--model", type=str, default="yolov8m-obb.pt")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=16)
args = parser.parse_args()

run_name = f"{args.model.split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
wandb.init(project="sentinel-kiln-repro", name=run_name, config=vars(args))

model = YOLO(args.model)
model.train(data=args.data_yaml, epochs=args.epochs, batch=args.batch, imgsz=640, project="sentinel-kiln-repro", name=run_name, amp=False)
wandb.finish()