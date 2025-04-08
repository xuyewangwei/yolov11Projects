# 如何训练模型

1.创建模型
```
from  ultralytics import YOLO
from ultralytics import settings

settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})

model = YOLO("./models/yolo11n.pt")
```
2.设置参数训练模型

```
results = model.train(data='coco8.yaml',epochs=100) # 这里调用coco8的现有数据库
```
3.评估模型准确率
```
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
```

