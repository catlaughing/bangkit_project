from flask import Flask, request, render_template, send_from_directory
import os
from preprocessing import preprocessing_image, get_encoding, create_model
from inference import predict_captions
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'static/img/uploaded'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])

model = create_model()
resnet = load_model('models/resnet50.h5')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET','POST'])
def hello_world():

    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        print(request.files)
        target = os.path.join(APP_ROOT,'static/img/uploaded')
        if 'file' not in request.files:
            print('file not uploaded')
            return render_template('main.html')
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            dest = 'static/img/uploaded/'+filename
            file.stream.seek(0)
            file.save(dest)
            print(dest)
            file.stream.seek(0)
            image = preprocessing_image(dest)
            encoded_image = get_encoding(resnet, image)
            caption = predict_captions(encoded_image,model)
            return render_template('result.html', caption=caption.capitalize() ,image_file=dest)


if __name__ == '__main__':
    app.run(debug=True)
    
