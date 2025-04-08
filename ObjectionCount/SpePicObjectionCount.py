import cv2
from ultralytics import YOLO
from ultralytics import settings
from ultralytics import solutions

model_name = "yolo11n.pt"
settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})
settings.update({"runs_dir":f"./run/{model_name}"})

def Objection_detect_area(img,region_points,model_name,class_to_count):
    # 和不特定的几乎一样，但是要指定训练模型yaml中的类
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_name,classes=class_to_count)
    result = counter(img) 
    return result #通过results.plot_im读取位置信息

img = cv2.imread("000000000009.jpg")
class_to_count = [0,2]
result = Objection_detect_area(img,[(20, 400), (1080, 400)],model_name,class_to_count)
# 这个模型中有如下代表的类
#  0: person
#   1: bicycle
#   2: car
#   3: motorcycle
#   4: airplane
#   5: bus
#   6: train
cv2.imwrite("output.jpg",result.plot_im)
