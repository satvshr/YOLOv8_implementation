from ultralytics import YOLO
import json

model = YOLO('yolov8n.pt') 
source = 'static/clip2.mp4'

results = model(source, stream=True) 

max_count = 0

for result in results:
    x = result.tojson()
    y = json.loads(x)
    count = 0
    for i in y:
        if i['name'] == "person":
            count += 1
    if count > max_count:
        max_count = count

print(max_count)










# # Define path to video file
# source = 'path/to/video.mp4'

# # Run inference on the source
# results = model(source, stream=True)  # generator of Results objects