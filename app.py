import os
import time
from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import shutil

# Initialize Flask app
app = Flask(__name__)

# Define the upload and results folder path
UPLOAD_FOLDER = 'static/uploads/'
RESULTS_FOLDER = 'static/results/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO('v18.pt')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has a file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return "No selected file"
        
        # If the file is valid
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Start model processing
            start_time = time.time()
            results = model.predict(source=upload_path, imgsz=640, conf=0.5, save=True)
            end_time = time.time()

            # Get the path of the result image saved by YOLO
            latest_run_folder = os.path.join('runs', 'detect', sorted(os.listdir('runs/detect'))[-1])
            result_image_path = os.path.join(latest_run_folder, filename)

            # Move result image to RESULTS_FOLDER
            new_result_image_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
            if os.path.exists(result_image_path):
                shutil.copy(result_image_path, new_result_image_path)

            execution_time = end_time - start_time

            # Return the result to the front-end
            return render_template('index.html', uploaded_image=upload_path, prediction_image=new_result_image_path, execution_time=execution_time)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
