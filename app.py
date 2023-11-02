from flask import Flask, render_template, request
from ultralytics import YOLO
import json, os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = YOLO('yolov8n.pt') 

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    video = request.files["video"]
    source = secure_filename(video.filename)
    print(type(source))
    print(source)
    source = os.path.join("C:\\Users\\satvm\\yolov8_objdetection\\proj\\static", source)
    print(type(source))
    print(source)
    if source:
        results = model(source, stream=True) 
        lst = []
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
            lst.append(count)
        avg = round(sum(lst) / len(lst), 1)
        pred = [max_count, avg]
        print(max_count)
        return render_template("index.html", prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)