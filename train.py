from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 可选但推荐，防止未来打包 exe 出问题

    # 1. 加载预训练模型
    model = YOLO(r"D:\app\yolo_project\model\cube-11s.pt")

    # 2. 开始微调训练
    model.train(
        data=r"D:\app\yolo_project\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,          # 关键：Windows 下先设为 0，避免多进程启动错误
        name="cube_finetune",
        exist_ok=True
    )