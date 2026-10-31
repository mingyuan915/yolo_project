from ultralytics import YOLO
from collections import Counter

# 1. 加载模型 (请根据你实际的路径修改)
model_path = r"D:\app\edge-lab\yolo\cube-11s.pt" 
model = YOLO(model_path)

# 打印模型支持的类别，确认包含蓝、红、黄、绿
print("模型支持的类别:", model.names)

# 2. 指定你要测试的图片
image_path = r"D:\\app\\yolo_project\\dataset\\images\\val"  # 请根据实际路径修改

# 3. 开始推理
# 注意：这里去掉了 classes 参数，默认检测所有它认识的类别
results = model.predict(source=image_path, conf=0.6, save=True)

# 4. 解析结果并统计
print("\n--- 检测结果 ---")
found_objects = []

for result in results:
    boxes = result.boxes
    for box in boxes:
        # 获取类别 ID 和名称
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        # 获取置信度
        confidence = float(box.conf[0])
        
        found_objects.append(class_name)
        print(f"✅ 发现 {class_name} 积木！置信度: {confidence:.2f}")

# 5. 统计每种颜色的数量 (使用 Counter)
if found_objects:
    counts = Counter(found_objects)
    print("\n--- 颜色统计 ---")
    for color, count in counts.items():
        print(f"{color} 积木: {count} 个")
else:
    print("未检测到任何积木。")

print(f"\n推理完成！带有框选结果的图片已保存在: D:\\app\\yolo_project\\runs\\detect\\predict")