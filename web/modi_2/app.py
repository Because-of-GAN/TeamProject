# -*- coding: utf-8 -*-
import os
import base64
from io import BytesIO
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from flask import Flask, render_template, request, session, redirect, url_for
from flask_dropzone import Dropzone
from model import Generator

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.add_url_rule('/sketch/<path:filename>', endpoint='sketch', view_func=app.send_static_file)
app.add_url_rule('/output/<path:filename>', endpoint='output', view_func=app.send_static_file)
########################################################################################################################
base_dir = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(base_dir, 'static/output')
upload_dir = os.path.join(base_dir, 'static/uploads')
sketch_dir = os.path.join(base_dir, 'static/sketch')
########################################################################################################################
app.config.update(
    UPLOADED_PATH=upload_dir,
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_REDIRECT_VIEW='result',  # set redirect view
    DROPZONE_DEFAULT_MESSAGE='Please Upload Your Image',
)
########################################################################################################################
dropzone = Dropzone(app)
########################################################################################################################
G = Generator()
G.load_state_dict(torch.load("./hed_G_500.pth", map_location=torch.device('cpu')))


########################################################################################################################


def removeAllFile(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file)


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        filenames = []
        for key, f in request.files.items():
            if key.startswith('file'):
                f.save(os.path.join(upload_dir, f.filename))
                filenames.append(f'{f.filename}')
        else:
            session['filenames'] = filenames
    else:
        removeAllFile(upload_dir)
        removeAllFile(output_dir)
        removeAllFile(sketch_dir)
    return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if not session.get('filenames', None):
        return redirect('/')
    filenames = []
    for e, i in enumerate(session.pop('filenames')):
        img2 = os.path.join(upload_dir, i)
        img = cv2.imread(img2)
        blur2 = cv2.bilateralFilter(img, 9, 180, 180)

        edge = cv2.Canny(blur2, 50, 200)
        cv2.imwrite(os.path.join(sketch_dir, f'sketch_{i}'), edge)
        filenames.append(f'sketch_{i}')
        session['sketch_filenames'] = filenames

    # for e, i in enumerate(session.pop('filenames')):
    #     img2 = os.path.join(upload_dir, i)
    #     img = cv2.imread(img2)
    #
    #     edge = cv2.Canny(img, 50, 200)
    #     edge = cv2.bitwise_not(edge)
    #     cv2.imwrite(os.path.join(sketch_dir, f'pri_sketch_{i}'), edge)
    #     filenames.append(f'pri_sketch_{i}')
    #     session['sketch_filenames'] = filenames
    return redirect(url_for('gen'))


def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


@app.route('/gen', methods=['POST', 'GET'])
def gen():
    if not session.get('sketch_filenames', None):
        print('no', session['sketch_filenames'])
        return redirect('/')
    filenames = []
    for e, i in enumerate(session.pop('sketch_filenames')):
        facade_a = Image.open(os.path.join(sketch_dir, sorted(os.listdir(sketch_dir))[e])).convert('RGB')
        facade_a = facade_a.resize((256, 256), Image.BICUBIC)
        facade_a = transforms.ToTensor()(facade_a)  # Quiz
        facade_a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(facade_a)
        fake_facade = G(facade_a.expand(1, 3, 256, 256))
        fake_facade = denorm(fake_facade.squeeze())

        plt.figure(figsize=(30, 90))
        plt.subplot(133)
        plt.imshow(to_data(fake_facade).numpy().transpose(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        file = os.path.join(output_dir, f'fake{e}.jpg')
        plt.savefig(file, bbox_inches="tight")
        filenames.append(f'fake{e}.jpg')
    return render_template('gen.html', filenames=filenames)


@app.route('/report', methods=['POST', 'GET'])
def report():
    return render_template('report.html')


@app.route('/painting', methods=['POST', 'GET'])
def painting():
    if request.method == 'POST':
        file = request.form['url']
        print(file)
        starter = file.find(',')
        image_data = file[starter + 1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        im.save(os.path.join(upload_dir, 'image.png'))

        filenames = []
        session['sketch_filenames'] = filenames

        return redirect(url_for('gen'))
    else:
        session['filenames'] = None
        session['sketch_filenames'] = None
        removeAllFile(upload_dir)
        removeAllFile(output_dir)
        removeAllFile(sketch_dir)
    return render_template('painting.html')


if __name__ == '__main__':
    app.run(debug=True)
