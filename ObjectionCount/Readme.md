# 环境要求
pip install opencv-python
pip install ultralytics

# 简介
使用cv2读取视频流，并实现指定区域内的技术

分为两个程序，一个是针对特定物体，一个是不针对特定物体进行计数

## 不针对特定物体进行计数

1.使用识别模型
```
import cv2
from ultralytics import YOLO
from ultralytics import settings
from ultralytics import solutions

model_name = "yolo11n.pt"
settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})
settings.update({"runs_dir":f"./run/{model_name}"})
```
2.定义图片识别函数
```这里注意solutions.ObjectCounter调用的是模型名字
def Objection_detect_area(img,region_points,model_name):
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_name)
    result = counter(img) 
    return result #通过results.plot_im读取位置信息
```
3.调用函数识别并打印
```
img = cv2.imread("000000000009.jpg")
result = Objection_detect_area(img,[(20, 400), (1080, 400)],model_name)

cv2.imwrite("output.jpg",result.plot_im)
```
