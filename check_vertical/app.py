from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from werkzeug.utils import secure_filename
from detection import detect_rebar_angles, detect_all_green_rebars

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # result_image, median_angle, _ = detect_rebar_angles(filepath)
            result_image, rebar_info, _ = detect_all_green_rebars(filepath)
            result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
            cv2.imwrite(result_path, result_image)

            return render_template('result.html', filename=filename, angle=rebar_info)

    return render_template('index.html')

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, port=5001)
