# 前面还是一样加载模型
```
from  ultralytics import YOLO
from ultralytics import settings

settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})


model = YOLO("./models/yolo11n.pt")
```

# 结果框输出
```
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
```
