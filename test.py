from ultralytics import YOLO

model = YOLO(r"D:\\app\\yolo_project\\runs\detect\\cube_finetune\\weights\\best.pt")
model.predict(
    source=r"D:\\app\\yolo_project\\dataset\\images\\val",  # 放几张新图
    save=True,
    conf=0.54,
    project=r"D:\\app\\yolo_project\\runs\\predict",
    name="cube_test"
)