import os

from flask import Flask, request, render_template
from flask_gtts import gtts
from tensorflow.keras.models import load_model

from inference import createCaption
from preprocessing import preprocessing_image, get_encoding

app = Flask(__name__)
gtts(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'static/img/uploaded'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])

model = load_model('models/model_rn50_glove.h5')
resnet = load_model('models/resnet50.h5')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('main.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('main.html')
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename.replace(' ','_')
            dest = 'static/img/uploaded/'+filename
            file.stream.seek(0)
            file.save(dest)
            file.stream.seek(0)
            image = preprocessing_image(dest)
            encoded_image = get_encoding(resnet, image)
            caption = createCaption(encoded_image,model)
            return render_template('result.html', caption=caption.capitalize() ,image_file=dest)


if __name__ == '__main__':
    app.run(debug=False)
