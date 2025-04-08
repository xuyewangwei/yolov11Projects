import cv2
from ultralytics import solutions

model_path = "./models/yolo11n.pt"
video_path = "./testSource/video_path"

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(),"Error Reading the video"

cropper = solutions.ObjectCropper(show = True,model_path = model_path,classes=[0,2])

while cap.isOpened():
    success,img = cap.read()
    if not success:
        print("end of video")
        break
    
    result = cropper(img)
    #print(result.plot_im)
    #cv2.imwrite("output_path",result.plot_im)

cap.release()
cv2.destroyAllWindows()
    
