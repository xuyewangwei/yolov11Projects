# 一共这些关键点位
鼻子
左眼
右眼
左耳
右耳
左肩
右肩
左肘
右肘
左腕
右手腕
左髋关节
右髋关节
左膝
右膝盖
左脚踝
右脚踝

# 使用模型如下
YOLO11n-pose	640	50.0	81.0	52.4 ± 0.5	1.7 ± 0.0	2.9	7.6
YOLO11s-pose	640	58.9	86.3	90.5 ± 0.6	2.6 ± 0.0	9.9	23.2
YOLO11m-pose	640	64.9	89.4	187.3 ± 0.8	4.9 ± 0.1	20.9	71.7
YOLO11l-pose	640	66.1	89.9	247.7 ± 1.1	6.4 ± 0.1	26.2	90.7
YOLO11x-pose	640	69.5	91.1	488.0 ± 13.9	12.1 ± 0.2	58.8	203.3

# 导入头文件
from ultralytics import YOLO

# 创建模型
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# 训练
results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)

# 评估
metrics = model.val()  # no arguments needed, dataset and settings remembered

# 运行
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# 导出
model.export(format="onnx")
