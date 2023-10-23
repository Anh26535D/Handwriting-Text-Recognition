from flask import Flask, render_template, request, send_from_directory
from crnn_model.Predictor import Predictor
import os

weight_path = r".\crnn_model\crnn_weights.h5"
max_len = 24
image_width, image_height = 256, 64

app = Flask(__name__)
predictor = Predictor(weight_path=weight_path, max_len=max_len,
                      image_width=image_width, image_height=image_height)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            path="./uploads/{}".format(file.filename)
            text = predictor.predict(path)
            print(f"[SERVER] The prediction is: {text}")
            return render_template("index.html", prediction=text, image=file.filename)
    
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    folder_path = r"E:\fpt_python_machine_learning\PML\final_project\app\uploads"
    file_list = os.listdir(folder_path)
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    app.run(debug=True)

