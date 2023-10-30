from ultralytics import YOLO
import json

model = YOLO('yolov8n.pt') 

results = model(['static/pic2.jpg', 'static/pic3.jpg'])  
 
for result in results:
    x = result.tojson()
    y = json.loads(x)
    count = 0
    for i in y:
        if i['name'] == "person":
            count += 1
    print(count)










# # Define path to video file
# source = 'path/to/video.mp4'

# # Run inference on the source
# results = model(source, stream=True)  # generator of Results objects