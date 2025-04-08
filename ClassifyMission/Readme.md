# 使用模型：
YOLO11n-cls	224	70.0	89.4	5.0 ± 0.3	1.1 ± 0.0	1.6	0.5
YOLO11s-cls	224	75.4	92.7	7.9 ± 0.2	1.3 ± 0.0	5.5	1.6
YOLO11m-cls	224	77.3	93.9	17.2 ± 0.4	2.0 ± 0.0	10.4	5.0
YOLO11l-cls	224	78.3	94.3	23.2 ± 0.3	2.8 ± 0.0	12.9	6.2
YOLO11x-cls	224	79.5	94.9	41.4 ± 0.9	3.8 ± 0.0	28.4	13.7

# 头文件
from ultralytics import YOLO

# 创建模型
model = YOLO("./models/yolo11s-cls.pt")

# 训练任务
results = model.train(data="mnist160", epochs=100, imgsz=64)

# 评估任务
metrics = model.val()  # no arguments needed, dataset and settings remembered

# 预测任务
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# 导出任务
# Export the model
model.export(format="abcd.pt")

