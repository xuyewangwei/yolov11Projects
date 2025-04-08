import cv2
from ultralytics import YOLO
from ultralytics import settings
from ultralytics import solutions

model_name = "yolo11n.pt"
settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})
settings.update({"runs_dir":f"./run/{model_name}"})


def Video_Objection_Count(video_path,output_video_path,region_points,model_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    counter=solutions.ObjectCounter(show = True,region = region_points,model = model_path)
    
    while cap.isOpened():
        success,img = cap.read()
        if not success:
            print("video frame is empty or process is complete")
            break
        result = counter(img)
        video_writer.write(result.plot_im)
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

region_points = [(20, 400), (1080, 400)]
video_path = "video_path"
output_path = "output_path"
Video_Objection_Count(video_path=video_path,output_video_path=output_path,region_points=region_points,model_path=model_name)
