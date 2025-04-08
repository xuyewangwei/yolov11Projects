# 首先下载预训练模型
使用wget
```
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt
```
### 这里有一些模型，可以选择下载使用
<https://docs.ultralytics.com/tasks/segment/#models>



# 接下里导入头文件与设置工作文件夹
```
from  ultralytics import YOLO
from ultralytics import settings

settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})
settings.update({"runs_dir":"./run"})
```

# 导入模型
```
model_name = "yolo11s-seg.pt"
model = YOLO(f"./models/{model_name}")
```

# 训练模型
```
# Train the model
results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
```



