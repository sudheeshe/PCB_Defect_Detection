from flask import Flask,request,render_template,redirect, url_for
import os
from yolov5.detect import run
from werkzeug.utils import secure_filename
import shutil


app = Flask(__name__)

# The image which gets through UI wille get saved at this location
app.config["IMAGE_UPLOADS"] = "D:/Ineuron/Project_workshop/pcb_inferencing/yolov5/static/Images"

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]


@app.route('/',methods = ["GET","POST"])
def upload_image():

	weight_path = r"runs/train/yolov5s_results7/weights/best.pt"
	conf = 0.3
	save_img = r'runs/detect'

	if request.method == "POST":

		shutil.rmtree('static/Images')
		os.mkdir('static/Images')

		image = request.files['file']

		filename = secure_filename(image.filename)

		basedir = os.path.abspath(os.path.dirname(__file__))
		image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))

		run(weights=weight_path,
			source=os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename),
			project='static/Images',
			name='exp',
			conf_thres=conf)

		return render_template("index.html", filename=filename)



	return render_template('index.html')


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename = "/Images/exp" + "prediction_" + filename), code=301)


app.run(debug=True,port=2000)