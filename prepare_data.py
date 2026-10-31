import os
import json
import shutil
from huggingface_hub import snapshot_download
from PIL import Image

# ================= 配置区 =================
REPO_ID = "edgeimpulse/cubes-on-conveyor-belt"
RAW_DIR = "D:/app/raw_edgeimpulse"                         # 原始下载目录（建议放外面）
YOLO_DIR = r"D:\app\yolo_project\dataset"                  # ★ 修改为你想要的 YOLO 数据集位置
HF_ENDPOINT = "https://hf-mirror.com"                      # 使用镜像加速

# 类别映射：你可以按需修改 ID 顺序
CLASS_MAP = {
    "blue": 0,
    "red": 1,
    "green": 2,
    "yellow": 3
}
# ==========================================

def download_dataset():
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    print("正在从 Hugging Face 镜像站下载原始数据集...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=RAW_DIR,
        allow_patterns=["*.jpg", "*.png", "*.jpeg", "*.labels"]
    )
    print("原始数据集下载完成。")

def convert_to_yolo():
    # 1. 创建 YOLO 标准目录结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)

    # 2. 定位 Edge Impulse 的子目录
    sub_dirs = [d for d in os.listdir(RAW_DIR) 
                if os.path.isdir(os.path.join(RAW_DIR, d))]

    total_converted = 0

    for sub_dir in sub_dirs:
        sub_path = os.path.join(RAW_DIR, sub_dir)
        labels_file = os.path.join(sub_path, "bounding_boxes.labels")

        if not os.path.exists(labels_file):
            print(f"跳过 {sub_dir}：未找到 bounding_boxes.labels 文件")
            continue

        with open(labels_file, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        if sub_dir.lower() == "training":
            split = "train"
        elif sub_dir.lower() == "testing":
            split = "val"
        else:
            split = "train"
            
        files_data = []
        if "boundingBoxes" in label_data and isinstance(label_data["boundingBoxes"], dict):
            for img_name, boxes in label_data["boundingBoxes"].items():
                files_data.append({"path": img_name, "boundingBoxes": boxes})
        elif "files" in label_data and isinstance(label_data["files"], list):
            files_data = label_data["files"]
        else:
            print(f"警告：{labels_file} 格式未知，跳过。")
            continue

        for file_info in files_data:
            img_name = file_info.get("path")
            if not img_name: continue
            
            src_img_path = os.path.join(sub_path, img_name)
            if not os.path.exists(src_img_path):
                src_img_path = os.path.join(sub_path, os.path.basename(img_name))
                if not os.path.exists(src_img_path):
                    continue

            with Image.open(src_img_path) as img:
                img_w, img_h = img.size

            yolo_lines = []
            for box in file_info.get("boundingBoxes", []):
                label_name = box.get("label", "").lower()
                if label_name not in CLASS_MAP: continue
                
                class_id = CLASS_MAP[label_name]
                x_abs, y_abs = box.get("x", 0), box.get("y", 0)
                w_abs, h_abs = box.get("width", 0), box.get("height", 0)

                x_center = (x_abs + w_abs / 2.0) / img_w
                y_center = (y_abs + h_abs / 2.0) / img_h
                w_norm, h_norm = w_abs / img_w, h_abs / img_h

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            if yolo_lines:
                base_name = os.path.splitext(os.path.basename(src_img_path))[0]
                unique_base = f"{sub_dir}_{base_name}"
                
                dst_img_path = os.path.join(YOLO_DIR, "images", split, f"{unique_base}.jpg")
                dst_txt_path = os.path.join(YOLO_DIR, "labels", split, f"{unique_base}.txt")

                # 直接复制图片
                shutil.copy2(src_img_path, dst_img_path)
                
                with open(dst_txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))

                total_converted += 1

    print(f"\n转换完成！共处理了 {total_converted} 张图片。")
    print(f"YOLO 数据集已保存在: {YOLO_DIR}")

if __name__ == "__main__":
    download_dataset()
    convert_to_yolo()