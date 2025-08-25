from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8x-seg.pt")

    model.train(
        data="dataset_merged/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="merged",
        project="runs/train",
        plots=True
    )

if __name__ == "__main__":
    train_model()
