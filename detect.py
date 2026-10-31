from ultralytics import YOLO
import os

# 1. 加载你下载好的模型 (请确认路径是否正确)
model_path = "D:\\app\\edge-lab\\yolo\\cube-11s.pt"
model = YOLO(model_path)

# 打印一下模型能识别的类别，确认包含 red
print("模型支持的类别:", model.names)

# 2. 指定你要测试的图片
image_path = "D:\\app\\yolo_project\\test.jpg"




# 3. 开始推理
# conf=0.5 表示置信度大于50%的才会被框出来
results = model.predict(source=image_path, conf=0.5, save=True)

# 4. 解析结果，看看有没有框到红积木
for result in results:
    boxes = result.boxes
    for box in boxes:
        # 获取类别 ID
        class_id = int(box.cls[0])
        # 获取类别名称 (比如 'red', 'blue')
        class_name = model.names[class_id]
        # 获取置信度
        confidence = float(box.conf[0])
        
        if class_name == 'red':
            print(f"✅ 发现红积木！置信度: {confidence:.2f}")
        else:
            print(f"发现其他积木: {class_name}，置信度: {confidence:.2f}")

print(f"\n推理完成！带有框选结果的图片已保存。")