import cv2
from ultralytics import YOLO
from ultralytics import settings
from ultralytics import solutions

model_name = "yolo11n.pt"
settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})
settings.update({"runs_dir":f"./run/{model_name}"})


def Objection_detect_area(img,region_points,model_name):
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_name)
    result = counter(img) 
    return result #通过results.plot_im读取位置信息

img = cv2.imread("000000000009.jpg")
result = Objection_detect_area(img,[(20, 400), (1080, 400)],model_name)

cv2.imwrite("output.jpg",result.plot_im)
