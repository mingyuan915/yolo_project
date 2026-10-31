# YOLO 积木检测项目

这是一个基于 [Ultralytics YOLO](https://docs.ultralytics.com/) 的目标检测小项目，用于识别图片中的积木颜色，并输出检测结果与可视化框选图片。

## 项目功能

- 加载本地训练好的 YOLO 模型进行图片推理。
- 检测图片中的积木类别，例如 `red`、`blue`、`yellow`、`green`。
- 输出每个目标的类别和置信度。
- 自动保存带框的检测结果图片。
- 支持统计图片中各类积木的数量。

## 文件说明

- `detect.py`：对单张图片进行检测，并重点判断是否识别到 `red` 类别。
- `detect_all.py`：对单张图片进行全面检测，并统计所有检测到的类别数量。
- `test.jpg`：示例输入图片。
- `runs/`：YOLO 推理后自动生成的结果目录。

## 环境依赖

本项目使用 Python 环境运行，主要依赖如下：

- `ultralytics`
- `torch`
- `opencv-python`

如果你的环境还没有安装依赖，可以在虚拟环境中执行：

```bash
pip install ultralytics torch opencv-python
或者 同样使用清华源来加速，避免网络超时。
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 虚拟环境激活

本项目已经包含一个虚拟环境目录 `yolo_env`，在 Windows 下推荐先激活它，再运行脚本。

### PowerShell 激活

在项目根目录执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\yolo_env\Scripts\Activate.ps1
```

如果激活成功，命令行前面通常会出现 `(yolo_env)`。

### 命令提示符 CMD 激活

如果你使用的是 CMD，可以执行：

```bat
yolo_env\Scripts\activate.bat
```

### 直接使用虚拟环境中的 Python

如果你不想手动激活，也可以直接调用虚拟环境里的解释器运行脚本：

```powershell
& .\yolo_env\Scripts\python.exe detect.py
& .\yolo_env\Scripts\python.exe detect_all.py
```

## 使用方法

### 1. 准备模型

确保代码里的模型路径正确。当前脚本使用的是类似下面的路径：

```text
D:\app\edge-lab\yolo\cube-11s.pt
```

如果你的模型文件位置不同，请修改 `detect.py` 和 `detect_all.py` 中的 `model_path`。

### 2. 准备图片

把待检测图片放到项目目录下，或者修改脚本中的 `image_path` 指向你的图片文件。

### 3. 运行脚本

检测单张图片并重点识别红色积木：

```bash
python detect.py
```

统计所有积木类别并输出数量：

```bash
python detect_all.py
```

## 输出结果

运行后会在控制台打印：

- 模型支持的类别名称。
- 每个检测目标的类别与置信度。
- 如果检测到 `red`，会单独提示。
- `detect_all.py` 会额外输出每种颜色的统计数量。

同时，YOLO 会把带检测框的结果图片保存到 `runs/detect/` 目录下。

## 代码逻辑简述

### `detect.py`

1. 加载训练好的 YOLO 模型。
2. 读取 `test.jpg`。
3. 执行推理，置信度阈值为 `0.5`。
4. 遍历检测框，判断类别是否为 `red`。
5. 打印检测结果并保存可视化图片。

### `detect_all.py`

1. 加载训练好的 YOLO 模型。
2. 读取 `test.jpg`。
3. 执行推理，置信度阈值为 `0.35`。
4. 收集所有检测到的类别。
5. 使用 `Counter` 统计每种颜色出现的次数。
6. 打印结果并保存可视化图片。

## 注意事项

- 模型路径必须存在，否则程序无法正常启动。
- 图片路径需要与实际文件位置一致。
- 如果检测结果不理想，可以适当调整 `conf` 参数。

## 适合的扩展方向

- 批量检测文件夹内所有图片。
- 把检测结果导出为 JSON 或 CSV。
- 增加摄像头实时识别功能。
- 在检测到特定颜色时触发告警或控制逻辑。
