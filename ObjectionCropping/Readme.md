# 对象剪裁
使用检测模型

1.导入使用的库
```
import cv2
from ultralytics import solutions

model_path = "./models/yolo11n.pt"
video_path = "./testSource/video_path"
```

2.初始化视频流和cropper模型，传入想选取的对象编号
```
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(),"Error Reading the video"

cropper = solutions.ObjectCropper(show = True,model_path = model_path,classes=[0,2])
```

3.对视频的每一帧进行运行
```
while cap.isOpened():
    success,img = cap.read()
    if not success:
        print("end of video")
        break
    
    result = cropper(img)
    #print(result.plot_im)
    #cv2.imwrite("output_path",result.plot_im)
```

4.释放连接，收尾
```
cap.release()
cv2.destroyAllWindows()
```

