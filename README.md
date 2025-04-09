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
# 设置参数训练模型

```
results = model.train(data='coco8.yaml',epochs=100) # 这里调用coco8的现有数据库
```
# 评估模型准确率
```
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
```
# 导出模型
```
model.export(format="engine", int8=True) # 创建engine文件
model.export(format="onnx") # 创建onnx文件
```

# 使用engine文件
```
# Load the exported TensorRT model
tensorrt_model = YOLO("yolo11n.engine")

results = tensorrt_model("https://ultralytics.com/images/bus.jpg")
```

```
# Load the exported ONNX model
onnx_model = YOLO("yolo11n.onnx")
results = onnx_model("https://ultralytics.com/images/bus.jpg")
```
# 两种文件的不同
<http://docs.ultralytics.com/zh/integrations/onnx/#cpu-deployment>
<https://docs.ultralytics.com/zh/integrations/tensorrt/#usage>
ONNX 模型通常用于 CPU，但也可部署在以下平台上：

GPU 加速：ONNX 完全支持GPU 加速，尤其是NVIDIA CUDA 。这样就能在NVIDIA GPU 上高效执行需要高计算能力的任务。

边缘和移动设备：ONNX 可扩展到边缘和移动设备，非常适合在设备上进行实时推理。它重量轻，与边缘硬件兼容。

网络浏览器：ONNX 可直接在网络浏览器中运行，为基于网络的交互式动态人工智能应用提供动力。

使用YOLO 和TensorRT INT8 的优势
减少模型大小：从 FP32 到 INT8 的量化可将模型大小减少 4 倍（在磁盘或内存中），从而加快下载速度，降低存储要求，并在部署模型时减少内存占用。

功耗更低：与 FP32 模型相比，INT8 导出的YOLO 模型降低了精度操作，因此功耗更低，尤其适用于电池供电的设备。

提高推理速度： TensorRT 可针对目标硬件优化模型，从而提高 GPU、嵌入式设备和加速器的推理速度。
