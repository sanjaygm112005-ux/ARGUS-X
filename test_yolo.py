from ultralytics import YOLO
import torch

print("Loading YOLO11n...")
model = YOLO("yolo11n.pt")

print("Running detection...")
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    conf=0.4,
)

print("Detection completed!")
print("MPS available:", torch.backends.mps.is_available())

for result in results:
    for box in result.boxes:
        print("Detected:", model.names[int(box.cls[0])])
