# yolov11Projects
盘点yolov11的基本操作,对应基本操作在对应文件夹中：
* 检测
* 分段
* 分类
* 识别
* 姿势
* obb
以及opencv 处理视频，图片的基本操作
yolov11进阶解决方案：
* 物体计数
* 
<https://docs.ultralytics.com/quickstart/>

# 环境配置
# Install the ultralytics package from PyPI
pip install ultralytics

# 参数配置
```打印当前配置
from ultralytics import settings

# 训练
results = model.train(data='coco8.yaml',epochs=100) # 这里调用coco8的现有数据库

# 评估模型
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
# 导出模型
model.export(format="njdwa.pt")

# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]
```

```使用update更改配置
from ultralytics import settings

# Update a setting
settings.update({"runs_dir": "/path/to/runs"})

# Update multiple settings
settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

# Reset settings to default values
settings.reset()
```

```以下是可以更改的配置
Name	Example Value	Data Type	Description
settings_version	'0.0.4'	str	Ultralytics settings version (distinct from the Ultralytics pip version)
datasets_dir	'/path/to/datasets'	str	Directory where datasets are stored
weights_dir	'/path/to/weights'	str	Directory where model weights are stored
runs_dir	'/path/to/runs'	str	Directory where experiment runs are stored
uuid	'a1b2c3d4'	str	Unique identifier for the current settings
sync	True	bool	Option to sync analytics and crashes to Ultralytics HUB
api_key	''	str	Ultralytics HUB API Key
clearml	True	bool	Option to use ClearML logging
comet	True	bool	Option to use Comet ML for experiment tracking and visualization
dvc	True	bool	Option to use DVC for experiment tracking and version control
hub	True	bool	Option to use Ultralytics HUB integration
mlflow	True	bool	Option to use MLFlow for experiment tracking
neptune	True	bool	Option to use Neptune for experiment tracking
raytune	True	bool	Option to use Ray Tune for hyperparameter tuning
tensorboard	True	bool	Option to use TensorBoard for visualization
wandb	True	bool	Option to use Weights & Biases logging
vscode_msg	True	bool	When a VS Code terminal is detected, enables a prompt to download the Ultralytics-Snippets extension.
```

# Python 运行yolov11
> 快速开始

### 导入头文件
> from ultralytics import YOLO

### 可以设置setting
```
settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})
```

### 创建模型
三种方式创建
```
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
```

### 运行模型
```
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
```

### 返回参数示例
```
# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
```

